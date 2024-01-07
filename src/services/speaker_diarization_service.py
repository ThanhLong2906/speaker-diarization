import os, shutil
# import wget
import json
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import get_uniqname_from_filepath
# from embedding_sound_cp import (
#     embedding_human_voice,
#     embedding_sound,
#     sound_similarity
# )
import librosa
import soundfile as sf
from typing import Any, Union
import logging

MODEL_CONFIG = os.path.join("../../input_config",'diar_infer_telephonic.yaml')
if not os.path.exists(MODEL_CONFIG):
    # config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    # MODEL_CONFIG = wget.download(config_url,"input_config")
    logging.error("there's no config file.")

def remove_folder(*args):
    for arg in args:
        if os.path.isdir(arg):
            shutil.rmtree(arg)
        else:
            logging.error(f"there is no folder: {arg}")


def audio_processing(audio_path, target_sr: int=16000, mono: bool=True, factor: float=1.0):
    # audio_file = os.path.join(output, get_uniqname_from_filepath(audio_path), "_processed.wav")
    y, sr = librosa.load(audio_path, mono=False)

    # convert from stereo to mono
    if mono:
        y = librosa.to_mono(y)
        
    # resample 
    if sr != target_sr:
        y = librosa.resample(y, orig_sr = sr, target_sr=target_sr)

    # volumn up factor time
    y = y * factor

    # write new audio file
    sf.write(audio_path, y, 
            target_sr, 'PCM_24')


def speaker_diarization(audio_file: Union[str, None] = None):
    audio_dir = os.path.dirname(audio_file)

    #create file input_manifest.json 
    meta = {
        'audio_filepath': audio_file, #args.input_audio, 
        'offset': 0, 
        'duration':None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': 2, 
        'rttm_filepath': None,#phone_rttm, 
        'uem_filepath' : None
    }

    with open(os.path.join(audio_dir,'input_manifest.json'),'w') as fp:
        json.dump(meta,fp)
        fp.write('\n')

    # create output folder (delete after promgram finish)
    output_dir = os.path.join(audio_dir, 'output')
    os.makedirs(output_dir,exist_ok=True)

    name = get_uniqname_from_filepath(audio_file)
    pred_rttm = os.path.join(output_dir,f'pred_rttms/{name}.rttm')

    # add information for config file for clustering process
    config = OmegaConf.load(MODEL_CONFIG)

    config.diarizer.manifest_filepath = os.path.join(audio_dir, 'input_manifest.json')
    config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
    # pretrained_vad = 'vad_multilingual_marblenet'
    # pretrained_speaker_model = 'titanet_large'
    # config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [5.0, 4.5, 4.0, 3.5, 3] #[1.5,1.25,1.0,0.75,0.5] 
    config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [2.5, 2.25, 2.0, 1.75, 1.5]#[0.75,0.625,0.5,0.375,0.1] 
    config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
    config.diarizer.oracle_vad = False # ----> ORACLE VAD 
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    # config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.1
    config.diarizer.vad.parameters.offset = 0.1
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.vad.parameters.window_length_in_sec = 0.2
    config.diarizer.vad.parameters.shift_length_in_sec = 0.05

    # Neural diarization
    # config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic' # Telephonic speaker diarization model 
    config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0]

    # data processsing
    audio_processing(audio_file, factor=2)

    # diarization
    # from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    # system_vad_msdd_model = NeuralDiarizer(cfg=config)
    # system_vad_msdd_model.diarize()

    from nemo_from_scratch.vad import VAD
    from nemo_from_scratch.speaker_embedding import SpeakerEmbedding
    from nemo_from_scratch.clustering import ClusterEmbedding
    from nemo_from_scratch.msdd import NeuralDiarizer

    a = VAD(config)
    a._perform_speech_activity_detection()
    _speaker_manifest_path = a._speaker_manifest_path
    
    b = SpeakerEmbedding(config)
    embs_and_timestamps = b.diarizer(_speaker_manifest_path)
    
    c = ClusterEmbedding(config)
    emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict = c.prepare_cluster_embs_infer(embs_and_timestamps=embs_and_timestamps)

    d = NeuralDiarizer(config)
    d.diarize(emb_sess_test_dict=emb_sess_avg_dict, emb_seq_test= emb_scale_seq_dict, clus_test_label_dict=base_clus_label_dict)

    del(d)
    del(a)
    del(b)
    del(c)
    del(emb_sess_avg_dict)     
    del(emb_scale_seq_dict)  
    del(base_clus_label_dict)     
    del(embs_and_timestamps)   

    return pred_rttm

