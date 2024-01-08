# from nemo.collections.asr.parts.utils.speaker_utils import (
#     audio_rttm_map,
#     get_embs_and_timestamps,
#     get_uniqname_from_filepath,
#     parse_scale_configs,
#     perform_clustering,
#     segments_manifest_to_subsegments_manifest,
#     validate_vad_manifest,
#     write_rttm2manifest,
# )
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    perform_clustering,
    get_embs_and_timestamps, 
    get_id_tup_dict,
    get_scale_mapping_argmat,
    get_uniq_id_list_from_manifest,
    labels_to_pyannote_object,
    make_rttm_with_overlap,
    parse_scale_configs,
    rttm_to_labels,
)
from nemo.collections.asr.metrics.der import score_labels
from typing import (Union, List, Any, Dict)
import os
from omegaconf import DictConfig, OmegaConf
from nemo_from_scratch.speaker_embedding import SpeakerEmbedding
from nemo.utils import logging, model_utils
import torch
import json
import pickle as pkl
import numpy as np
from statistics import mode

class ClusteringDiarizer:
    def __init__(self, cfg: Union[DictConfig, Any], speaker_model=None):
        if isinstance(cfg, DictConfig):
            cfg = model_utils.convert_model_config_to_dict_config(cfg)
            # Convert config to support Hydra 1.0+ instantiation
            cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg

        self._out_dir = self._cfg.diarizer.out_dir
        # Diarizer set up
        self._diarizer_params = self._cfg.diarizer 

        # Clustering params
        self._cluster_params = self._diarizer_params.clustering.parameters

        self.multiscale_args_dict = parse_scale_configs(
            self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
            self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )

    def diarize(self, embs_and_timestamps, paths2audio_files: List[str] = None, batch_size: int = 0):
        """
        Diarize files provided through paths2audio_files or manifest file
        input:
        paths2audio_files (List[str]): list of paths to file containing audio file
        batch_size (int): batch_size considered for extraction of speaker embeddings and VAD computation
        """
        if batch_size:
            self._cfg.batch_size = batch_size

        self.AUDIO_RTTM_MAP = audio_rttm_map(self._diarizer_params.manifest_filepath)

        out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(out_rttm_dir, exist_ok=True)
        # Clustering
        all_reference, all_hypothesis = perform_clustering(
            embs_and_timestamps=embs_and_timestamps, # output of SpeakerEmbedding.diarizer
            AUDIO_RTTM_MAP=self.AUDIO_RTTM_MAP,
            out_rttm_dir=out_rttm_dir,
            clustering_params=self._cluster_params,
            device=torch.device, # add urself
            verbose=self.verbose,
        )
        logging.info("Outputs are saved in {} directory".format(os.path.abspath(self._diarizer_params.out_dir)))

        # Scoring
        return score_labels(
            self.AUDIO_RTTM_MAP,
            all_reference,
            all_hypothesis,
            collar=self._diarizer_params.collar,
            ignore_overlap=self._diarizer_params.ignore_overlap,
            verbose=self.verbose,
        )
    
    @property
    def verbose(self) -> bool:
        return self._cfg.verbose

class ClusterEmbedding(torch.nn.Module):
    def __init__(
        self, cfg_diar_infer: DictConfig):
        super().__init__()
        self.cfg_diar_infer = cfg_diar_infer
        # self._cfg_msdd = cfg_msdd_model
        # self._speaker_model = speaker_model
        self.scale_window_length_list = list(
            self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.window_length_in_sec
        )
        self.scale_n = len(self.scale_window_length_list)
        self.base_scale_index = len(self.scale_window_length_list) - 1
        self.clus_diar_model = ClusteringDiarizer(cfg=self.cfg_diar_infer)
    
    def get_scale_map(self, embs_and_timestamps):
        """
        Save multiscale mapping data into dictionary format.

        Args:
            embs_and_timestamps (dict):
                Dictionary containing embedding tensors and timestamp tensors. Indexed by `uniq_id` string.
        Returns:
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.
        """
        session_scale_mapping_dict = {}
        for uniq_id, uniq_embs_and_timestamps in embs_and_timestamps.items():
            scale_mapping_dict = get_scale_mapping_argmat(uniq_embs_and_timestamps)
            session_scale_mapping_dict[uniq_id] = scale_mapping_dict
        return session_scale_mapping_dict
    
    def load_emb_scale_seq_dict(self, out_dir):
        """
        Load saved embeddings generated by clustering diarizer. This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where embedding pickle files are saved.
        Returns:
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
        """
        window_len_list = list(self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.window_length_in_sec)
        emb_scale_seq_dict = {scale_index: None for scale_index in range(len(window_len_list))}
        for scale_index in range(len(window_len_list)):
            pickle_path = os.path.join(
                out_dir, 'speaker_outputs', 'embeddings', f'subsegments_scale{scale_index}_embeddings.pkl'
            )
            logging.info(f"Loading embedding pickle file of scale:{scale_index} at {pickle_path}")
            with open(pickle_path, "rb") as input_file:
                emb_dict = pkl.load(input_file)
            for key, val in emb_dict.items():
                emb_dict[key] = val
            emb_scale_seq_dict[scale_index] = emb_dict
        return emb_scale_seq_dict
    
    def check_clustering_labels(self, out_dir):
        """
        Check whether the laoded clustering label file is including clustering results for all sessions.
        This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where clustering result files are saved.
        Returns:
            file_exists (bool):
                Boolean that indicates whether clustering result file exists.
            clus_label_path (str):
                Path to the clustering label output file.
        """
        clus_label_path = os.path.join(
            out_dir, 'speaker_outputs', f'subsegments_scale{self.base_scale_index}_cluster.label'
        )
        file_exists = os.path.exists(clus_label_path)
        if not file_exists:
            logging.info(f"Clustering label file {clus_label_path} does not exist.")
        return file_exists, clus_label_path
    
    def load_clustering_labels(self, out_dir):
        """
        Load clustering labels generated by clustering diarizer. This function is used for inference mode of MSDD.

        Args:
            out_dir (str):
                Path to the directory where clustering result files are saved.
        Returns:
            emb_scale_seq_dict (dict):
                List containing clustering results in string format.
        """
        file_exists, clus_label_path = self.check_clustering_labels(out_dir)
        logging.info(f"Loading cluster label file from {clus_label_path}")
        with open(clus_label_path) as f:
            clus_labels = f.readlines()
        return clus_labels
    
    def assign_labels_to_longer_segs(self, base_clus_label_dict: Dict, session_scale_mapping_dict: Dict):
        """
        In multi-scale speaker diarization system, clustering result is solely based on the base-scale (the shortest scale).
        To calculate cluster-average speaker embeddings for each scale that are longer than the base-scale, this function assigns
        clustering results for the base-scale to the longer scales by measuring the distance between subsegment timestamps in the
        base-scale and non-base-scales.

        Args:
            base_clus_label_dict (dict):
                Dictionary containing clustering results for base-scale segments. Indexed by `uniq_id` string.
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.

        Returns:
            all_scale_clus_label_dict (dict):
                Dictionary containing clustering labels of all scales. Indexed by scale_index in integer format.

        """
        all_scale_clus_label_dict = {scale_index: {} for scale_index in range(self.scale_n)}
        for uniq_id, uniq_scale_mapping_dict in session_scale_mapping_dict.items():
            base_scale_clus_label = np.array([x[-1] for x in base_clus_label_dict[uniq_id]])
            all_scale_clus_label_dict[self.base_scale_index][uniq_id] = base_scale_clus_label
            for scale_index in range(self.scale_n - 1):
                new_clus_label = []
                assert (
                    uniq_scale_mapping_dict[scale_index].shape[0] == base_scale_clus_label.shape[0]
                ), "The number of base scale labels does not match the segment numbers in uniq_scale_mapping_dict"
                max_index = max(uniq_scale_mapping_dict[scale_index])
                for seg_idx in range(max_index + 1):
                    if seg_idx in uniq_scale_mapping_dict[scale_index]:
                        seg_clus_label = mode(base_scale_clus_label[uniq_scale_mapping_dict[scale_index] == seg_idx])
                    else:
                        seg_clus_label = 0 if len(new_clus_label) == 0 else new_clus_label[-1]
                    new_clus_label.append(seg_clus_label)
                all_scale_clus_label_dict[scale_index][uniq_id] = new_clus_label
        return all_scale_clus_label_dict

    def get_cluster_avg_embs(
        self, emb_scale_seq_dict: Dict, clus_labels: List, speaker_mapping_dict: Dict, session_scale_mapping_dict: Dict
    ):
        """
        MSDD requires cluster-average speaker embedding vectors for each scale. This function calculates an average embedding vector for each cluster (speaker)
        and each scale.

        Args:
            emb_scale_seq_dict (dict):
                Dictionary containing embedding sequence for each scale. Keys are scale index in integer.
            clus_labels (list):
                Clustering results from clustering diarizer including all the sessions provided in input manifest files.
            speaker_mapping_dict (dict):
                Speaker mapping dictionary in case RTTM files are provided. This is mapping between integer based speaker index and
                speaker ID tokens in RTTM files.
                Example:
                    {'en_0638': {'speaker_0': 'en_0638_A', 'speaker_1': 'en_0638_B'},
                     'en_4065': {'speaker_0': 'en_4065_B', 'speaker_1': 'en_4065_A'}, ...,}
            session_scale_mapping_dict (dict):
                Dictionary containing multiscale mapping information for each session. Indexed by `uniq_id` string.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing speaker mapping information and cluster-average speaker embedding vector.
                Each session-level dictionary is indexed by scale index in integer.
            output_clus_label_dict (dict):
                Subegmentation timestamps in float type and Clustering result in integer type. Indexed by `uniq_id` keys.
        """
        self.scale_n = len(emb_scale_seq_dict.keys())
        emb_sess_avg_dict = {
            scale_index: {key: [] for key in emb_scale_seq_dict[self.scale_n - 1].keys()}
            for scale_index in emb_scale_seq_dict.keys()
        }
        output_clus_label_dict, emb_dim = self.get_base_clus_label_dict(clus_labels, emb_scale_seq_dict)
        all_scale_clus_label_dict = self.assign_labels_to_longer_segs(
            output_clus_label_dict, session_scale_mapping_dict
        )
        for scale_index in emb_scale_seq_dict.keys():
            for uniq_id, _emb_tensor in emb_scale_seq_dict[scale_index].items():
                if type(_emb_tensor) == list:
                    emb_tensor = torch.tensor(np.array(_emb_tensor))
                else:
                    emb_tensor = _emb_tensor
                clus_label_list = all_scale_clus_label_dict[scale_index][uniq_id]
                spk_set = set(clus_label_list)

                # Create a label array which identifies clustering result for each segment.
                label_array = torch.Tensor(clus_label_list)
                avg_embs = torch.zeros(emb_dim, self.max_num_speakers)
                for spk_idx in spk_set:
                    selected_embs = emb_tensor[label_array == spk_idx]
                    avg_embs[:, spk_idx] = torch.mean(selected_embs, dim=0)

                if speaker_mapping_dict is not None:
                    inv_map = {clus_key: rttm_key for rttm_key, clus_key in speaker_mapping_dict[uniq_id].items()}
                else:
                    inv_map = None

                emb_sess_avg_dict[scale_index][uniq_id] = {'mapping': inv_map, 'avg_embs': avg_embs}
        return emb_sess_avg_dict, output_clus_label_dict
    def get_base_clus_label_dict(self, clus_labels: List[str], emb_scale_seq_dict: Dict[int, dict]):
        """
        Retrieve base scale clustering labels from `emb_scale_seq_dict`.

        Args:
            clus_labels (list):
                List containing cluster results generated by clustering diarizer.
            emb_scale_seq_dict (dict):
                Dictionary containing multiscale embedding input sequences.
        Returns:
            base_clus_label_dict (dict):
                Dictionary containing start and end of base scale segments and its cluster label. Indexed by `uniq_id`.
            emb_dim (int):
                Embedding dimension in integer.
        """
        base_clus_label_dict = {key: [] for key in emb_scale_seq_dict[self.base_scale_index].keys()}
        for line in clus_labels:
            uniq_id = line.split()[0]
            label = int(line.split()[-1].split('_')[-1])
            stt, end = [round(float(x), 2) for x in line.split()[1:3]]
            base_clus_label_dict[uniq_id].append([stt, end, label])
        emb_dim = emb_scale_seq_dict[0][uniq_id][0].shape[0]
        return base_clus_label_dict, emb_dim
    
    def run_clustering_diarizer(self, embs_and_timestamps):
        """
        If no pre-existing data is provided, run clustering diarizer from scratch. This will create scale-wise speaker embedding
        sequence, cluster-average embeddings, scale mapping and base scale clustering labels. Note that speaker embedding `state_dict`
        is loaded from the `state_dict` in the provided MSDD checkpoint.

        Args:
            manifest_filepath (str):
                Input manifest file for creating audio-to-RTTM mapping.
            emb_dir (str):
                Output directory where embedding files and timestamp files are saved.

        Returns:
            emb_sess_avg_dict (dict):
                Dictionary containing cluster-average embeddings for each session.
            emb_scale_seq_dict (dict):
                Dictionary containing embedding tensors which are indexed by scale numbers.
            base_clus_label_dict (dict):
                Dictionary containing clustering results. Clustering results are cluster labels for the base scale segments.
        """
        # self.cfg_diar_infer.diarizer.manifest_filepath = manifest_filepath
        # self.cfg_diar_infer.diarizer.out_dir = emb_dir

        # Run ClusteringDiarizer which includes system VAD or oracle VAD.
        self._out_dir = self.clus_diar_model._diarizer_params.out_dir
        self.out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(self.out_rttm_dir, exist_ok=True)

        self.clus_diar_model._cluster_params = self.cfg_diar_infer.diarizer.clustering.parameters
        self.clus_diar_model.multiscale_args_dict[
            "multiscale_weights"
        ] = self.cfg_diar_infer.diarizer.speaker_embeddings.parameters.multiscale_weights
        self.clus_diar_model._diarizer_params.speaker_embeddings.parameters = (
            self.cfg_diar_infer.diarizer.speaker_embeddings.parameters
        )
        cluster_params = self.clus_diar_model._cluster_params
        cluster_params = dict(cluster_params) if isinstance(cluster_params, DictConfig) else cluster_params.dict()
        clustering_params_str = json.dumps(cluster_params, indent=4)

        logging.info(f"Multiscale Weights: {self.clus_diar_model.multiscale_args_dict['multiscale_weights']}")
        logging.info(f"Clustering Parameters: {clustering_params_str}")
        scores = self.clus_diar_model.diarize(batch_size=self.cfg_diar_infer.batch_size, embs_and_timestamps=embs_and_timestamps)

        # If RTTM (ground-truth diarization annotation) files do not exist, scores is None.
        if scores is not None:
            metric, speaker_mapping_dict, _ = scores
        else:
            metric, speaker_mapping_dict = None, None

        # Get the mapping between segments in different scales.
        # self._embs_and_timestamps = get_embs_and_timestamps(
        #     embs_and_timestamps, self.clus_diar_model.multiscale_args_dict
        # )
        session_scale_mapping_dict = self.get_scale_map(embs_and_timestamps)
        emb_scale_seq_dict = self.load_emb_scale_seq_dict(self.cfg_diar_infer.diarizer.out_dir)
        clus_labels = self.load_clustering_labels(self.cfg_diar_infer.diarizer.out_dir)
        emb_sess_avg_dict, base_clus_label_dict = self.get_cluster_avg_embs(
            emb_scale_seq_dict, clus_labels, speaker_mapping_dict, session_scale_mapping_dict
        )
        emb_scale_seq_dict['session_scale_mapping'] = session_scale_mapping_dict
        return emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict, metric

    def prepare_cluster_embs_infer(self, embs_and_timestamps):
        """
        Launch clustering diarizer to prepare embedding vectors and clustering results.
        """
        self.max_num_speakers = self.cfg_diar_infer.diarizer.clustering.parameters.max_num_speakers
        self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict, _ = self.run_clustering_diarizer(embs_and_timestamps)
        return self.emb_sess_test_dict, self.emb_seq_test, self.clus_test_label_dict
    
    @property
    def verbose(self) -> bool:
        return self._cfg.verbose
    
def main():
    input_config = "./input_config/diar_infer_telephonic.yaml"
    cfg = OmegaConf.load(input_config)
    a = SpeakerEmbedding(cfg)
    embs_and_timestamps = a.diarizer()
    # b = ClusteringDiarizer(cfg)
    # b.diarize(embs_and_timestamps)
    b = ClusterEmbedding(cfg)
    emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict = b.prepare_cluster_embs_infer(embs_and_timestamps=embs_and_timestamps)

if __name__=='__main__':
    main()