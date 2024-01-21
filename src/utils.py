# import cv2
# import numpy as np
# import time
import json
import os
# import wget
import logging
import librosa
import soundfile as sf
import torch
# def resize_image(img, max_pixels):
#     if img.shape[0] * img.shape[1] > max_pixels:
#         k = np.sqrt(max_pixels / (img.shape[0] * img.shape[1]))
#         return cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)))
#     else:
#         return img

# class TimeCounter(object):
#     def __init__(self, task_name) -> None:
#         self.task_name = task_name
#         print(f"==> Start {self.task_name}...")
#         self.start_time = time.time()

#     def done(self):
#         print("==> Done {} | Time: {}".format(self.task_name, time.time() - self.start_time))

def get_config_path():
    MODEL_CONFIG = os.path.join("../input_config",'diar_infer_telephonic.yaml')
    if not os.path.exists(MODEL_CONFIG):
        # config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
        # MODEL_CONFIG = wget.download(config_url,"input_config")
        logging.error("there's no config file.")

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

def convert2wav(file_path):
    format = file_path.split(".")[1]
    try:
        wav_path = file_path.split(".")[0] + ".wav"
        from pydub import AudioSegment
        if format == "mp3":
            sound = AudioSegment.from_mp3(file_path)
        else: 
            sound = AudioSegment.from_file(file_path, format=format)
        sound.export(wav_path, format="wav")
        return wav_path
    except:
        logging.error(f"Not support {format} format")

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
    
def audio_trimming(src_path, des_path, time: list):
    from pydub import AudioSegment
    # t1 = t1 * 1000 #Works in milliseconds
    # t2 = t2 * 1000
    newAudio = AudioSegment.from_wav(src_path)
    newAudio = newAudio[time[0]:time[1]]
    newAudio.export(des_path, format="wav")

def get_available_model_names(class_name):
    "lists available pretrained model names from NGC"
    available_models = class_name.list_available_models()
    return list(map(lambda x: x.pretrained_model_name, available_models))

def json_split(json_filepath: str, chunk_size: int):
    list_of_filename = []
    with open(json_filepath, 'r') as f:
        j = 0
        lines = f.readlines()
        if len(lines) <= chunk_size:
            list_of_filename.append(json_filepath)
        else:
            for i in range(chunk_size, len(lines), chunk_size):
                with open(json_filepath.split(".")[0] + f"_{i}" + ".json", "+w") as nf:
                    list_of_filename.append(json_filepath.split(".")[0] + f"_{i}" + ".json")
                    for m in range(j, i):
                        json.dump(json.loads(lines[m]), nf)
                        nf.write('\n')
                j = i
            if len(lines)%chunk_size != 0:
                # write the residual of jsonfile
                with open(json_filepath.split(".")[0] + f"_{len(lines)}" + ".json", "+w") as nf:
                    list_of_filename.append(json_filepath.split(".")[0] + f"_{len(lines)}" + ".json")
                    for i in range(j, len(lines)):
                        json.dump(json.loads(lines[i]), nf)
                        nf.write('\n')
    return list_of_filename

def get_available_model_names(class_name):
    "lists available pretrained model names from NGC"
    available_models = class_name.list_available_models()
    return list(map(lambda x: x.pretrained_model_name, available_models))

