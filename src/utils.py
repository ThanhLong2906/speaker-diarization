# import cv2
# import numpy as np
# import time
import json
import os
# import wget
import logging

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
        for i in range(chunk_size, len(lines), chunk_size):
            with open(json_filepath.split(".")[0] + f"_{i}" + ".json", "+w") as nf:
                list_of_filename.append(json_filepath.split(".")[0] + f"_{i}" + ".json")
                for m in range(j, i):
                    json.dump(json.loads(lines[m]), nf)
                    nf.write('\n')
            j = i
        if len(lines)%chunk_size != 0:
            # write the residual of jsonfile
            for i in range(j, len(lines)):
                with open(json_filepath.split(".")[0] + f"_{len(lines)}" + ".json", "+w") as nf:
                    list_of_filename.append(json_filepath.split(".")[0] + f"_{len(lines)}" + ".json")
                    json.dump(json.loads(lines[i]), nf)
                    nf.write('\n')
    return list_of_filename

