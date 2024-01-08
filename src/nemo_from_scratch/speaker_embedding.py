from typing import Union, Any, List, Dict
from configuration import Settings
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import shutil
from nemo.utils import logging, model_utils
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo_from_scratch.utils import get_available_model_names
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_embs_and_timestamps,
    get_uniqname_from_filepath,
    parse_scale_configs,
    perform_clustering,
    segments_manifest_to_subsegments_manifest,
    validate_vad_manifest,
    write_rttm2manifest,
)
import torch
from tqdm import tqdm
import json
import pickle as pkl 
import time
from nemo_from_scratch.vad import VAD
try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield

class SpeakerEmbedding:
    def __init__(self, cfg: Union[DictConfig, Any]):
        if isinstance(cfg, DictConfig):
            cfg = model_utils.convert_model_config_to_dict_config(cfg)
            # Convert config to support Hydra 1.0+ instantiation
            cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg

        # Diarizer set up
        self._diarizer_params = self._cfg.diarizer

        self._out_dir = self._diarizer_params.out_dir

        self._speaker_dir = os.path.join(self._diarizer_params.out_dir, 'speaker_outputs')

        if os.path.exists(self._speaker_dir):
            logging.warning("Deleting previous clustering diarizer outputs.")
            shutil.rmtree(self._speaker_dir, ignore_errors=True)
        os.makedirs(self._speaker_dir)

        # init speaker model
        self.multiscale_embeddings_and_timestamps = {}
        self._init_speaker_model()
        self._speaker_params = self._cfg.diarizer.speaker_embeddings.parameters
        # self._speaker_manifest_path = os.path.join(self._out_dir, 'vad_outputs', 'vad_out.json')
        # Clustering params
        # self._cluster_params = self._diarizer_params.clustering.parameters

    def embed_voice(self, path2audio: str):
        """
            args:
                path2audio: string path to audio file 
            return:
                embedding tensor
        """
        embedding = self._speaker_model.get_embedding(path2audio)
        return embedding
    
    def save_voice(self, path2audio: str, metadatas: List[Dict] | None, collection_name: Union[str, None] = None):
        from vector_store import Milvus
        # from utils import TimeCounter
        if collection_name is None:
            collection_name = "audio_db"

        self.app_settings = Settings()
        self.vector_store = Milvus(
            connection_args={"host": self.app_settings.MILVUS_HOST, "port": self.app_settings.MILVUS_PORT},
            collection_name=collection_name
        )
        embedding = np.array(self.embed_voice(path2audio))
        self.vector_store.add_embeddings(embedding, metadatas, timeout=200)

    def _init_speaker_model(self, speaker_model=None):
        """
        Initialize speaker embedding model with model name or path passed through config
        """
        if speaker_model is not None:
            self._speaker_model = speaker_model
        else:
            model_path = self._cfg.diarizer.speaker_embeddings.model_path
        if model_path is not None and model_path.endswith('.nemo'):
            self._speaker_model = EncDecSpeakerLabelModel.restore_from(model_path, map_location=self._cfg.device)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            self._speaker_model = EncDecSpeakerLabelModel.load_from_checkpoint(
                model_path, map_location=self._cfg.device
            )
            logging.info("Speaker Model restored locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(EncDecSpeakerLabelModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "ecapa_tdnn"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self._speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                model_name=model_path, map_location=self._cfg.device
            )

        self.multiscale_args_dict = parse_scale_configs(
        self._diarizer_params.speaker_embeddings.parameters.window_length_in_sec,
        self._diarizer_params.speaker_embeddings.parameters.shift_length_in_sec,
        self._diarizer_params.speaker_embeddings.parameters.multiscale_weights,
        )
    
    def _setup_spkr_test_data(self, manifest_file):
        spk_dl_config = {
            'manifest_filepath': manifest_file,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': self._cfg.get('batch_size'),
            'trim_silence': False,
            'labels': None,
            'num_workers': self._cfg.num_workers,
        }
        self._speaker_model.setup_test_data(spk_dl_config)

    def _run_segmentation(self, window: float, shift: float, scale_tag: str = ''):

        self.subsegments_manifest_path = os.path.join(self._speaker_dir, f'subsegments{scale_tag}.json')
        logging.info(
            f"Subsegmentation for embedding extraction:{scale_tag.replace('_',' ')}, {self.subsegments_manifest_path}"
        )
        self.subsegments_manifest_path = segments_manifest_to_subsegments_manifest(
            segments_manifest_file=self._speaker_manifest_path,
            subsegments_manifest_file=self.subsegments_manifest_path,
            window=window,
            shift=shift,
        )
        return None
    
    def _extract_embeddings(self, manifest_file: str, scale_idx: int, num_scales: int):
        """
        This method extracts speaker embeddings from segments passed through manifest_file
        Optionally you may save the intermediate speaker embeddings for debugging or any use. 
        """
        logging.info("Extracting embeddings for Diarization")
        self._setup_spkr_test_data(manifest_file)
        self.embeddings = {}
        self._speaker_model.eval()
        self.time_stamps = {}

        all_embs = torch.empty([0])
        for test_batch in tqdm(
            self._speaker_model.test_dataloader(),
            desc=f'[{scale_idx+1}/{num_scales}] extract embeddings',
            leave=True,
            disable=not self.verbose,
        ):  
            test_batch = [x.to(self._speaker_model.device) for x in test_batch]
            audio_signal, audio_signal_len, labels, slices = test_batch
            # print(f"test batch: {test_batch}")
            # print(f"len test batch: {len(test_batch)}")
            # print(f"len audio_signal: {len(audio_signal)}")
            with autocast():
                _, embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                emb_shape = embs.shape[-1]
                embs = embs.view(-1, emb_shape)
                all_embs = torch.cat((all_embs, embs.cpu().detach()), dim=0)
                del embs
            del test_batch

        with open(manifest_file, 'r', encoding='utf-8') as manifest:
            for i, line in enumerate(manifest.readlines()):
                line = line.strip()
                dic = json.loads(line)
                uniq_name = get_uniqname_from_filepath(dic['audio_filepath'])
                if uniq_name in self.embeddings:
                    self.embeddings[uniq_name] = torch.cat((self.embeddings[uniq_name], all_embs[i].view(1, -1)))
                else:
                    self.embeddings[uniq_name] = all_embs[i].view(1, -1)
                if uniq_name not in self.time_stamps:
                    self.time_stamps[uniq_name] = []
                start = dic['offset']
                end = start + dic['duration']
                self.time_stamps[uniq_name].append([start, end])

        if self._speaker_params.save_embeddings:
            embedding_dir = os.path.join(self._speaker_dir, 'embeddings')
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir, exist_ok=True)

            prefix = get_uniqname_from_filepath(manifest_file)
            name = os.path.join(embedding_dir, prefix)
            self._embeddings_file = name + f'_embeddings.pkl'
            pkl.dump(self.embeddings, open(self._embeddings_file, 'wb'))
            logging.info("Saved embedding files to {}".format(embedding_dir))

        return self.embeddings
    
    def diarizer(self, _speaker_manifest_path):
        self._speaker_manifest_path = _speaker_manifest_path

        # Segmentation
        scales = self.multiscale_args_dict['scale_dict'].items()

        for scale_idx, (window, shift) in scales:
            # Segmentation for the current scale (scale_idx)
            self._run_segmentation(window, shift, scale_tag=f'_scale{scale_idx}')

            # Embedding Extraction for the current scale (scale_idx)
            self._extract_embeddings(self.subsegments_manifest_path, scale_idx, len(scales))
            self.multiscale_embeddings_and_timestamps[scale_idx] = [self.embeddings, self.time_stamps]
        embs_and_timestamps = get_embs_and_timestamps(
            self.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )
        return embs_and_timestamps

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
    
if __name__=='__main__':
    main()
