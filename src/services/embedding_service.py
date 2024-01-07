# import requests
import json
import os
# import pickle as pkl
from configuration import Settings
from typing import Dict, List, Union
import torch
import numpy as np
from omegaconf import DictConfig
from nemo.utils import logging, model_utils
from tqdm import tqdm
import logging
import wget
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_uniqname_from_filepath,
)
try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


MODEL_CONFIG = os.path.join("../../input_config",'diar_infer_telephonic.yaml')
if not os.path.exists(MODEL_CONFIG):
    # config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    # MODEL_CONFIG = wget.download(config_url,"input_config")
    logging.error("There's no config file.")

def sound_similarity(audio_embs, voice_emb):
    '''
        audio_sound_emb: link to pickle file of audio sound segment
        human_sound_emb: link to pickle file of human sound 
        return: cosin similarity of 2 sound
    '''
    # audio_emb = next(iter(pkl.load(open(audio_sound_emb, "rb")).values()))
    # human_emb = next(iter(pkl.load(open(human_sound_emb, "rb")).values()))
    result = []
    # print(f"length audio_embs: {audio_embs.shape}")
    if isinstance(audio_embs, dict):
        cos = torch.nn.CosineSimilarity(dim=0)
        for _, value in audio_embs.items():
            for idx in range(value.shape[0]):
                file_emb = value[idx][:]
                # print(f"shape value: {file_emb.squeeze().shape}")
                # print(f"shape voice_emb: {voice_emb.shape}")
                result.append(cos(file_emb.squeeze(), voice_emb))
    else: logging.error("audio_embs must be a dictionary")
    return result

class EmbeddingVoiceService(object):
    def __init__(self):
        from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel 
        cfg = OmegaConf.load(MODEL_CONFIG)
        if isinstance(cfg, DictConfig):
            cfg = model_utils.convert_model_config_to_dict_config(cfg)
            # Convert config to support Hydra 1.0+ instantiation
            cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.model = EncDecSpeakerLabelModel.restore_from(restore_path=self._cfg.diarizer.speaker_embeddings.embed_path, map_location=device)
        
    def embed_voice(self, path2audio: str):
        """
            args:
                path2audio: string path to audio file 
            return:
                embedding tensor
        """
        embedding = self.model.get_embedding(path2audio)
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
    
    def _setup_spkr_test_data(self, manifest_file):
        spk_dl_config = {
            'manifest_filepath': manifest_file,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': self._cfg.get('batch_size'),
            'trim_silence': False,
            'labels': "UNK",
            'num_workers': self._cfg.num_workers,
        }
        self.model.setup_test_data(spk_dl_config)

    def embeddings_from_manifest(self, manifest_file: str):
        """
        This method extracts speaker embeddings from segments passed through manifest_file
        Optionally you may save the intermediate speaker embeddings for debugging or any use. 
        """

        logging.info("Extracting embeddings for Diarization")
        self._setup_spkr_test_data(manifest_file)
        self.embeddings = {}
        self.model.eval()
        self.time_stamps = {}

        all_embs = torch.empty([0])
        for test_batch in tqdm(
            self.model.test_dataloader(),
            desc=f'extract embeddings',
            leave=True,
            disable=not self.verbose,
        ):
            test_batch = [x.to(self.model.device) for x in test_batch]
            audio_signal, audio_signal_len, labels, slices = test_batch
            with autocast():
                _, embs = self.model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                emb_shape = embs.shape[-1]
                embs = embs.view(-1, emb_shape)
                all_embs = torch.cat((all_embs, embs.cpu().detach()), dim=0)
            del test_batch

        with open(manifest_file, 'r', encoding='utf-8') as manifest:
            for i, line in enumerate(manifest.readlines()):
                line = line.strip()
                dic = json.loads(line)
                uniq_name = get_uniqname_from_filepath(dic['audio_filepath']) + f"_{i}"
                if uniq_name in self.embeddings:
                    self.embeddings[uniq_name] = torch.cat((self.embeddings[uniq_name], all_embs[i].view(1, -1)))
                else:
                    self.embeddings[uniq_name] = all_embs[i].view(1, -1)
                if uniq_name not in self.time_stamps:
                    self.time_stamps[uniq_name] = []
                start = dic['offset']
                end = start + dic['duration']
                self.time_stamps[uniq_name].append([start, end])
                label = dic['label']
                uniq_id = dic['uniq_id']

        return self.embeddings
    
    @property
    def verbose(self) -> bool:
        return self._cfg.verbose