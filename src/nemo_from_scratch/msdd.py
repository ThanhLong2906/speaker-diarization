import torch
from omegaconf import DictConfig, OmegaConf
from typing import (
    Optional,
    Dict, 
    List,
    Union,
    Tuple,
    Any
)
from pyannote.metrics.diarization import DiarizationErrorRate
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel
from nemo_from_scratch.utils import get_available_model_names
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_embs_and_timestamps,
    get_id_tup_dict,
    get_scale_mapping_argmat,
    get_uniq_id_list_from_manifest,
    labels_to_pyannote_object,
    make_rttm_with_overlap,
    parse_scale_configs,
    rttm_to_labels,
)
import os
from nemo.utils import logging
import pickle as pkl
import numpy as np
import json
from pytorch_lightning import LightningModule
from nemo.collections.asr.models.configs.diarizer_config import NeuralDiarizerInferenceConfig
import json
from nemo.collections.asr.metrics.der import score_labels
from tqdm import tqdm
from nemo_from_scratch.clustering import ClusterEmbedding
from nemo_from_scratch.speaker_embedding import SpeakerEmbedding
from nemo_from_scratch.vad import VAD
try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


class NeuralDiarizer(LightningModule):
    """
    Class for inference based on multiscale diarization decoder (MSDD). MSDD requires initializing clustering results from
    clustering diarizer. Overlap-aware diarizer requires separate RTTM generation and evaluation modules to check the effect of
    overlap detection in speaker diarization.
    """

    def __init__(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig]):
        super().__init__()
        self._cfg = cfg
        self._out_dir = self._cfg.diarizer.out_dir
        # Parameter settings for MSDD model
        self.use_speaker_model_from_ckpt = cfg.diarizer.msdd_model.parameters.get('use_speaker_model_from_ckpt', True)
        self.use_clus_as_main = cfg.diarizer.msdd_model.parameters.get('use_clus_as_main', False)
        self.max_overlap_spks = cfg.diarizer.msdd_model.parameters.get('max_overlap_spks', 2)
        self.num_spks_per_model = cfg.diarizer.msdd_model.parameters.get('num_spks_per_model', 2)
        self.use_adaptive_thres = cfg.diarizer.msdd_model.parameters.get('use_adaptive_thres', True)
        self.max_pred_length = cfg.diarizer.msdd_model.parameters.get('max_pred_length', 0)
        self.diar_eval_settings = cfg.diarizer.msdd_model.parameters.get(
            'diar_eval_settings', [(0.25, True), (0.25, False), (0.0, False)]
        )

        self._init_msdd_model(cfg)
        self.diar_window_length = cfg.diarizer.msdd_model.parameters.diar_window_length
        self.msdd_model.cfg = self.transfer_diar_params_to_model_params(self.msdd_model, cfg)

        # # Initialize clustering and embedding preparation instance (as a diarization encoder).
        # self.clustering_embedding = ClusterEmbedding(
        #     cfg_diar_infer=cfg, cfg_msdd_model=self.msdd_model.cfg, speaker_model=self._speaker_model
        # ) # output of clustering 
        # self.clustering_embedding = clustering

        # Parameters for creating diarization results from MSDD outputs.
        self.clustering_max_spks = self.msdd_model._cfg.max_num_of_spks
        self.overlap_infer_spk_limit = cfg.diarizer.msdd_model.parameters.get(
            'overlap_infer_spk_limit', self.clustering_max_spks
        )

    def _init_msdd_model(self, cfg: Union[DictConfig, NeuralDiarizerInferenceConfig]):

        """
        Initialized MSDD model with the provided config. Load either from `.nemo` file or `.ckpt` checkpoint files.
        """
        model_path = cfg.diarizer.msdd_model.model_path
        if model_path.endswith('.nemo'):
            logging.info(f"Using local nemo file from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.restore_from(restore_path=model_path, map_location=cfg.device)
        elif model_path.endswith('.ckpt'):
            logging.info(f"Using local checkpoint from {model_path}")
            self.msdd_model = EncDecDiarLabelModel.load_from_checkpoint(
                checkpoint_path=model_path, map_location=cfg.device
            )
        else:
            if model_path not in get_available_model_names(EncDecDiarLabelModel):
                logging.warning(f"requested {model_path} model name not available in pretrained models, instead")
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self.msdd_model = EncDecDiarLabelModel.from_pretrained(model_name=model_path, map_location=cfg.device)
        # Load speaker embedding model state_dict which is loaded from the MSDD checkpoint.
        if self.use_speaker_model_from_ckpt:
            self._speaker_model = self.extract_standalone_speaker_model()
        else:
            self._speaker_model = None

    def transfer_diar_params_to_model_params(self, msdd_model, cfg):
        """
        Transfer the parameters that are needed for MSDD inference from the diarization inference config files
        to MSDD model config `msdd_model.cfg`.
        """
        msdd_model.cfg.diarizer.out_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.batch_size = cfg.diarizer.msdd_model.parameters.infer_batch_size
        msdd_model.cfg.test_ds.seq_eval_mode = cfg.diarizer.msdd_model.parameters.seq_eval_mode
        msdd_model._cfg.max_num_of_spks = cfg.diarizer.clustering.parameters.max_num_speakers
        return msdd_model.cfg
    
    # def get_emb_clus_infer(self, cluster_embeddings):
    #     """Assign dictionaries containing the clustering results from the class instance `cluster_embeddings`.
    #     """
    #     self.msdd_model.emb_sess_test_dict = cluster_embeddings.emb_sess_test_dict
    #     self.msdd_model.clus_test_label_dict = cluster_embeddings.clus_test_label_dict
    #     self.msdd_model.emb_seq_test = cluster_embeddings.emb_seq_test
    def get_range_average(
        self, signals: torch.Tensor, emb_vectors: torch.Tensor, diar_window_index: int, test_data_collection: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        This function is only used when `split_infer=True`. This module calculates cluster-average embeddings for the given short range.
        The range length is set by `self.diar_window_length`, and each cluster-average is only calculated for the specified range.
        Note that if the specified range does not contain some speakers (e.g. the range contains speaker 1, 3) compared to the global speaker sets
        (e.g. speaker 1, 2, 3, 4) then the missing speakers (e.g. speakers 2, 4) are assigned with zero-filled cluster-average speaker embedding.

        Args:
            signals (Tensor):
                Zero-padded Input multi-scale embedding sequences.
                Shape: (length, scale_n, emb_vectors, emb_dim)
            emb_vectors (Tensor):
                Cluster-average multi-scale embedding vectors.
                Shape: (length, scale_n, emb_vectors, emb_dim)
            diar_window_index (int):
                Index of split diarization wondows.
            test_data_collection (collections.DiarizationLabelEntity)
                Class instance that is containing session information such as targeted speaker indices, audio filepath and RTTM filepath.

        Returns:
            return emb_vectors_split (Tensor):
                Cluster-average speaker embedding vectors for each scale.
            emb_seq (Tensor):
                Zero-padded multi-scale embedding sequences.
            seq_len (int):
                Length of the sequence determined by `self.diar_window_length` variable.
        """
        emb_vectors_split = torch.zeros_like(emb_vectors)
        uniq_id = os.path.splitext(os.path.basename(test_data_collection.audio_file))[0]
        clus_label_tensor = torch.tensor([x[-1] for x in self.msdd_model.clus_test_label_dict[uniq_id]])
        for spk_idx in range(len(test_data_collection.target_spks)):
            stt, end = (
                diar_window_index * self.diar_window_length,
                min((diar_window_index + 1) * self.diar_window_length, clus_label_tensor.shape[0]),
            )
            seq_len = end - stt
            if stt < clus_label_tensor.shape[0]:
                target_clus_label_tensor = clus_label_tensor[stt:end]
                emb_seq, seg_length = (
                    signals[stt:end, :, :],
                    min(
                        self.diar_window_length,
                        clus_label_tensor.shape[0] - diar_window_index * self.diar_window_length,
                    ),
                )
                target_clus_label_bool = target_clus_label_tensor == test_data_collection.target_spks[spk_idx]

                # There are cases where there is no corresponding speaker in split range, so any(target_clus_label_bool) could be False.
                if any(target_clus_label_bool):
                    emb_vectors_split[:, :, spk_idx] = torch.mean(emb_seq[target_clus_label_bool], dim=0)

                # In case when the loop reaches the end of the sequence
                if seq_len < self.diar_window_length:
                    emb_seq = torch.cat(
                        [
                            emb_seq,
                            torch.zeros(self.diar_window_length - seq_len, emb_seq.shape[1], emb_seq.shape[2]).to(
                                signals.device
                            ),
                        ],
                        dim=0,
                    )
            else:
                emb_seq = torch.zeros(self.diar_window_length, emb_vectors.shape[0], emb_vectors.shape[1]).to(
                    signals.device
                )
                seq_len = 0
        return emb_vectors_split, emb_seq, seq_len
    
    def get_range_clus_avg_emb(
        self, test_batch: List[torch.Tensor], _test_data_collection: List[Any], device: torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function is only used when `get_range_average` function is called. This module calculates cluster-average embeddings for
        the given short range. The range length is set by `self.diar_window_length`, and each cluster-average is only calculated for the specified range.

        Args:
            test_batch: (list)
                List containing embedding sequences, length of embedding sequences, ground truth labels (if exists) and initializing embedding vectors.
            test_data_collection: (list)
                List containing test-set dataloader contents. test_data_collection includes wav file path, RTTM file path, clustered speaker indices.

        Returns:
            sess_emb_vectors (Tensor):
                Tensor of cluster-average speaker embedding vectors.
                Shape: (batch_size, scale_n, emb_dim, 2*num_of_spks)
            sess_emb_seq (Tensor):
                Tensor of input multi-scale embedding sequences.
                Shape: (batch_size, length, scale_n, emb_dim)
            sess_sig_lengths (Tensor):
                Tensor of the actucal sequence length without zero-padding.
                Shape: (batch_size)
        """
        _signals, signal_lengths, _targets, _emb_vectors = test_batch
        sess_emb_vectors, sess_emb_seq, sess_sig_lengths = [], [], []
        split_count = torch.ceil(torch.tensor(_signals.shape[1] / self.diar_window_length)).int()
        self.max_pred_length = max(self.max_pred_length, self.diar_window_length * split_count)
        for k in range(_signals.shape[0]):
            signals, emb_vectors, test_data_collection = _signals[k], _emb_vectors[k], _test_data_collection[k]
            for diar_window_index in range(split_count):
                emb_vectors_split, emb_seq, seq_len = self.get_range_average(
                    signals, emb_vectors, diar_window_index, test_data_collection
                )
                sess_emb_vectors.append(emb_vectors_split)
                sess_emb_seq.append(emb_seq)
                sess_sig_lengths.append(seq_len)
        sess_emb_vectors = torch.stack(sess_emb_vectors).to(device)
        sess_emb_seq = torch.stack(sess_emb_seq).to(device)
        sess_sig_lengths = torch.tensor(sess_sig_lengths).to(device)
        return sess_emb_vectors, sess_emb_seq, sess_sig_lengths
    
    def diar_infer(
        self, test_batch: List[torch.Tensor], test_data_collection: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Launch forward_infer() function by feeding the session-wise embedding sequences to get pairwise speaker prediction values.
        If split_infer is True, the input audio clips are broken into short sequences then cluster average embeddings are calculated
        for inference. Split-infer might result in an improved results if calculating clustering average on the shorter tim-espan can
        help speaker assignment.

        Args:
            test_batch: (list)
                List containing embedding sequences, length of embedding sequences, ground truth labels (if exists) and initializing embedding vectors.
            test_data_collection: (list)
                List containing test-set dataloader contents. test_data_collection includes wav file path, RTTM file path, clustered speaker indices.

        Returns:
            preds (Tensor):
                Tensor containing predicted values which are generated from MSDD model.
            targets (Tensor):
                Tensor containing binary ground-truth values.
            signal_lengths (Tensor):
                The actual Session length (number of steps = number of base-scale segments) without zero padding.
        """
        signals, signal_lengths, _targets, emb_vectors = test_batch
        if self._cfg.diarizer.msdd_model.parameters.split_infer:
            split_count = torch.ceil(torch.tensor(signals.shape[1] / self.diar_window_length)).int()
            sess_emb_vectors, sess_emb_seq, sess_sig_lengths = self.get_range_clus_avg_emb(
                test_batch, test_data_collection, device=self.msdd_model.device
            )
            with autocast():
                _preds, scale_weights = self.msdd_model.forward_infer(
                    input_signal=sess_emb_seq,
                    input_signal_length=sess_sig_lengths,
                    emb_vectors=sess_emb_vectors,
                    targets=None,
                )
            _preds = _preds.reshape(len(signal_lengths), split_count * self.diar_window_length, -1)
            _preds = _preds[:, : signals.shape[1], :]
        else:
            with autocast():
                _preds, scale_weights = self.msdd_model.forward_infer(
                    input_signal=signals, input_signal_length=signal_lengths, emb_vectors=emb_vectors, targets=None
                )
        self.max_pred_length = max(_preds.shape[1], self.max_pred_length)
        preds = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
        targets = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
        preds[:, : _preds.shape[1], :] = _preds
        return preds, targets, signal_lengths
    
    def get_pred_mat(self, data_list: List[Union[Tuple[int], List[torch.Tensor]]]) -> torch.Tensor:
        """
        This module puts together the pairwise, two-speaker, predicted results to form a finalized matrix that has dimension of
        `(total_len, n_est_spks)`. The pairwise results are evenutally averaged. For example, in 4 speaker case (speaker 1, 2, 3, 4),
        the sum of the pairwise results (1, 2), (1, 3), (1, 4) are then divided by 3 to take average of the sigmoid values.

        Args:
            data_list (list):
                List containing data points from `test_data_collection` variable. `data_list` has sublists `data` as follows:
                data[0]: `target_spks` tuple
                    Examples: (0, 1, 2)
                data[1]: Tensor containing estimaged sigmoid values.
                   [[0.0264, 0.9995],
                    [0.0112, 1.0000],
                    ...,
                    [1.0000, 0.0512]]

        Returns:
            sum_pred (Tensor):
                Tensor containing the averaged sigmoid values for each speaker.
        """
        all_tups = tuple()
        for data in data_list:
            all_tups += data[0]
        n_est_spks = len(set(all_tups))
        digit_map = dict(zip(sorted(set(all_tups)), range(n_est_spks)))
        total_len = max([sess[1].shape[1] for sess in data_list])
        sum_pred = torch.zeros(total_len, n_est_spks)
        for (_dim_tup, pred_mat) in data_list:
            dim_tup = [digit_map[x] for x in _dim_tup]
            if len(pred_mat.shape) == 3:
                pred_mat = pred_mat.squeeze(0)
            if n_est_spks <= self.num_spks_per_model:
                sum_pred = pred_mat
            else:
                _end = pred_mat.shape[0]
                sum_pred[:_end, dim_tup] += pred_mat.cpu().float()
        sum_pred = sum_pred / (n_est_spks - 1)
        return sum_pred
    
    def get_integrated_preds_list(
        self, uniq_id_list: List[str], test_data_collection: List[Any], preds_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Merge multiple sequence inference outputs into a session level result.

        Args:
            uniq_id_list (list):
                List containing `uniq_id` values.
            test_data_collection (collections.DiarizationLabelEntity):
                Class instance that is containing session information such as targeted speaker indices, audio filepaths and RTTM filepaths.
            preds_list (list):
                List containing tensors filled with sigmoid values.

        Returns:
            output_list (list):
                List containing session-level estimated prediction matrix.
        """
        session_dict = get_id_tup_dict(uniq_id_list, test_data_collection, preds_list)
        output_dict = {uniq_id: [] for uniq_id in uniq_id_list}
        for uniq_id, data_list in session_dict.items():
            sum_pred = self.get_pred_mat(data_list)
            output_dict[uniq_id] = sum_pred.unsqueeze(0)
        output_list = [output_dict[uniq_id] for uniq_id in uniq_id_list]
        return output_list
    
    @torch.no_grad()
    def run_pairwise_diarization(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Setup the parameters needed for batch inference and run batch inference. Note that each sample is pairwise speaker input.
        The pairwise inference results are reconstructed to make session-wise prediction results.

        Returns:
            integrated_preds_list: (list)
                List containing the session-wise speaker predictions in torch.tensor format.
            targets_list: (list)
                List containing the ground-truth labels in matrix format filled with  0 or 1.
            signal_lengths_list: (list)
                List containing the actual length of each sequence in session.
        """
        self.out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms') #self.clustering_embedding.out_rttm_dir 
        self.msdd_model.setup_test_data(self.msdd_model.cfg.test_ds)
        self.msdd_model.eval()
        cumul_sample_count = [0]
        preds_list, targets_list, signal_lengths_list = [], [], []
        uniq_id_list = get_uniq_id_list_from_manifest(self.msdd_model.cfg.test_ds.manifest_filepath)
        test_data_collection = [d for d in self.msdd_model.data_collection]
        for sidx, test_batch in enumerate(tqdm(self.msdd_model.test_dataloader())):
            signals, signal_lengths, _targets, emb_vectors = test_batch
            cumul_sample_count.append(cumul_sample_count[-1] + signal_lengths.shape[0])
            preds, targets, signal_lengths = self.diar_infer(
                test_batch, test_data_collection[cumul_sample_count[-2] : cumul_sample_count[-1]]
            )
            if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
                self.msdd_model._accuracy_test(preds, targets, signal_lengths)

            preds_list.extend(list(torch.split(preds, 1)))
            targets_list.extend(list(torch.split(targets, 1)))
            signal_lengths_list.extend(list(torch.split(signal_lengths, 1)))

        if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
            f1_score, simple_acc = self.msdd_model.compute_accuracies()
            logging.info(f"Test Inference F1 score. {f1_score:.4f}, simple Acc. {simple_acc:.4f}")
        integrated_preds_list = self.get_integrated_preds_list(uniq_id_list, test_data_collection, preds_list)
        return integrated_preds_list, targets_list, signal_lengths_list
    
    def run_overlap_aware_eval(
        self, preds_list: List[torch.Tensor], threshold: float
    ) -> List[Optional[Tuple[DiarizationErrorRate, Dict]]]:
        """
        Based on the predicted sigmoid values, render RTTM files then evaluate the overlap-aware diarization results.

        Args:
            preds_list: (list)
                List containing predicted pairwise speaker labels.
            threshold: (float)
                A floating-point threshold value that determines overlapped speech detection.
                    - If threshold is 1.0, no overlap speech is detected and only detect major speaker.
                    - If threshold is 0.0, all speakers are considered active at any time step.
        """
        logging.info(
            f"     [Threshold: {threshold:.4f}] [use_clus_as_main={self.use_clus_as_main}] [diar_window={self.diar_window_length}]"
        )
        outputs = []
        manifest_filepath = self.msdd_model.cfg.test_ds.manifest_filepath
        rttm_map = audio_rttm_map(manifest_filepath)
        for k, (collar, ignore_overlap) in enumerate(self.diar_eval_settings):
            all_reference, all_hypothesis = make_rttm_with_overlap(
                manifest_filepath,
                self.msdd_model.clus_test_label_dict,
                preds_list,
                threshold=threshold,
                infer_overlap=True,
                use_clus_as_main=self.use_clus_as_main,
                overlap_infer_spk_limit=self.overlap_infer_spk_limit,
                use_adaptive_thres=self.use_adaptive_thres,
                max_overlap_spks=self.max_overlap_spks,
                out_rttm_dir=self.out_rttm_dir,
            )
            output = score_labels(
                rttm_map,
                all_reference,
                all_hypothesis,
                collar=collar,
                ignore_overlap=ignore_overlap,
                verbose=self._cfg.verbose,
            )
            outputs.append(output)
        logging.info(f"  \n")
        return outputs
    
    @torch.no_grad()
    def diarize(self, emb_sess_test_dict, emb_seq_test, clus_test_label_dict) -> Optional[List[Optional[List[Tuple[DiarizationErrorRate, Dict]]]]]:
        """
        Launch diarization pipeline which starts from VAD (or a oracle VAD stamp generation), initialization clustering and multiscale diarization decoder (MSDD).
        Note that the result of MSDD can include multiple speakers at the same time. Therefore, RTTM output of MSDD needs to be based on `make_rttm_with_overlap()`
        function that can generate overlapping timestamps. `self.run_overlap_aware_eval()` function performs DER evaluation.
        """
        # self.clustering_embedding.prepare_cluster_embs_infer()
        self.msdd_model.pairwise_infer = True

        # self.get_emb_clus_infer(self.clustering_embedding)
        self.msdd_model.emb_sess_test_dict = emb_sess_test_dict
        self.msdd_model.clus_test_label_dict = clus_test_label_dict
        self.msdd_model.emb_seq_test = emb_seq_test

        preds_list, targets_list, signal_lengths_list = self.run_pairwise_diarization()
        thresholds = list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold)
        return [self.run_overlap_aware_eval(preds_list, threshold) for threshold in thresholds]

    @property
    def verbose(self) -> bool:
        return self._cfg.verbose 
    
def main():
    input_config = "../../input_config/diar_infer_telephonic.yaml"
    cfg = OmegaConf.load(input_config)

    a = VAD(cfg)
    a._perform_speech_activity_detection()
    _speaker_manifest_path = a._speaker_manifest_path
    
    b = SpeakerEmbedding(cfg)
    embs_and_timestamps = b.diarizer(_speaker_manifest_path)
    
    c = ClusterEmbedding(cfg)
    emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict = c.prepare_cluster_embs_infer(embs_and_timestamps=embs_and_timestamps)

    d = NeuralDiarizer(cfg)
    d.diarize(emb_sess_test_dict=emb_sess_avg_dict, emb_seq_test= emb_scale_seq_dict, clus_test_label_dict=base_clus_label_dict)

if __name__=='__main__':
    main()