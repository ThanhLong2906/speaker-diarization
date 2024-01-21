from fastapi import FastAPI, HTTPException, UploadFile, Body, BackgroundTasks
# import shortuu
# from utils import TimeCounter
import os
# import time
from contextlib import asynccontextmanager
from pymilvus import connections, utility, Collection
from configuration import Settings
from typing import Union
from nemo.collections.asr.parts.utils.speaker_utils import get_uniqname_from_filepath, write_rttm2manifest
from fastapi.responses import JSONResponse
import logging
import torch
import numpy as np
from vector_store import Milvus
from omegaconf import OmegaConf
# import wget
from utils import audio_processing, sound_similarity, convert2wav
from nemo_functions.vad import VAD
from nemo_functions.speaker_embedding import SpeakerEmbedding
from nemo_functions.clustering import ClusterEmbedding
from nemo_functions.msdd import NeuralDiarizer
import json
import shutil
from pathlib import Path
MODEL_CONFIG = os.path.join("../input_config",'diar_infer_telephonic.yaml')
if not os.path.exists(MODEL_CONFIG):
    # config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    # MODEL_CONFIG = wget.download(config_url,"input_config")
    logging.error("there's no config file.")

tags_metadata = [
    {
        "name": "extraction_job",
        "description": "Operations with timestamp extraction jobs from audio",
    },
    {
        "name": "audio",
        "description": "Manage people audio",
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT, 
    )
    # reset cache
    yield

app = FastAPI(title="Extract information from video API", openapi_tags=tags_metadata, lifespan=lifespan)

@app.post("/add_voice", tags=["embedding_voice"], description="""
Add people voice to database

Params:
- audio file: 
- people_metadatas: list of metadata of the person in the audio
""")
async def add_voice(background_tasks: BackgroundTasks, voice: UploadFile, name: str, collection_name: Union[str, None]=None):
    # background_tasks.add_task(new_voice, voice, name, collection_name)
    new_voice(background_tasks, voice, name, collection_name)
    return {"message": "Voice uploaded, processing..."}


def new_voice(background_tasks: BackgroundTasks, voice: UploadFile, name: str, collection_name: Union[str, None]=None):
    if voice is None:
        raise HTTPException(status_code=400, detail="No audio file provided")
    if utility.has_collection(collection_name):
        col = Collection(name=collection_name)
        res = col.query(expr=f"name like '{name}'", output_fields=['pk'])
        del col
        if len(res) != 0:
            raise HTTPException(status_code=400, detail=f"name \'{name}\' exist.")
        
    background_tasks.add_task(run_add_voice, voice, name, collection_name)  

async def run_add_voice(voice: UploadFile, name: str, collection_name: Union[str, None]=None):
    content = await voice.read()
    # dot_index = voice.filename.rfind(".")
    # file_name = people_metadatas.people_metadatas[0].id + "-" + str(round(time.time())) + img.filename[dot_index:]
    # file_name = name + voice.filename[dot_index:]
    os.makedirs(f"{Settings().GLOBAL_VOICE_DIR}/{collection_name}",exist_ok=True)
    save_path = f"{Settings().GLOBAL_VOICE_DIR}/{collection_name}/{voice.filename}"
    with open(save_path, 'wb+') as f:
        f.write(content)
    save_path = convert2wav(save_path)
    audio_processing(save_path)
    # task.done()
    # laod config file
    cfg = OmegaConf.load(MODEL_CONFIG)
    # task = TimeCounter(task_name="Extracting faces and save to database")
    service = SpeakerEmbedding(cfg)
    metadata = [{"name":name, "source": voice.filename}]
    service.save_voice(save_path, metadata, collection_name=collection_name)
    # task.done()
    # if isinstance(background_tasks, BackgroundTasks):
    #  background_tasks.add_task(runProcessing, file_path, collection_name)
    # else:
    #     return runProcessing_Sync(file_path, collection_name)
    print(f"Add voice of {name} successfuly.")
    # return {"id": file_name}

@app.post("/speaker_diarization", tags=["voice"])
async def diarization(voice: UploadFile, name: str, collection_name: Union[str, None]=None):
    if voice is None:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # query vector by name in db
    db = Milvus(collection_name=collection_name)
    output_fields = ["vector"]
    res = db.col.query(expr=f"name like '{name}'", output_fields=output_fields)
    if len(res) != 1:
        logging.error(f"There is no person or more than one person named {name}")
        resp = {
                "speaker": None,
                "duration": None
            }
    else:
        voice_emb = torch.as_tensor(np.array(res[0]['vector']))

        # load config file
        cfg = OmegaConf.load(MODEL_CONFIG)
        # create temporary store (remove after promgram finish)
        tmp_dir = "temporary_store"
        os.makedirs(tmp_dir, exist_ok=True)

        # write copy of input audio file
        content = await voice.read()
        save_path = os.path.join(tmp_dir, voice.filename)
        with open(save_path, 'wb+') as f:
            f.write(content)
        save_path = convert2wav(save_path)
        audio_processing(save_path)
        
        # # diarization
        # pred_rttm_path = speaker_diarization(audio_file=save_path)

        audio_dir = os.path.dirname(save_path)

        #create file input_manifest.json 
        meta = {
            'audio_filepath': save_path, #args.input_audio, 
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

        name = get_uniqname_from_filepath(save_path)
        pred_rttm = os.path.join(output_dir,f'pred_rttms/{name}.rttm')

        # add information for config file for clustering process
        config = OmegaConf.load(MODEL_CONFIG)

        config.diarizer.manifest_filepath = os.path.join(audio_dir, 'input_manifest.json')
        config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
        

        # data processsing
        audio_processing(save_path, factor=2)

        a = VAD(config)
        a._perform_speech_activity_detection()
        _speaker_manifest_path = a._speaker_manifest_path
        
        b = SpeakerEmbedding(config)
        embs_and_timestamps = b.diarizer(_speaker_manifest_path)
        
        c = ClusterEmbedding(config)
        emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict = c.prepare_cluster_embs_infer(embs_and_timestamps=embs_and_timestamps)

        d = NeuralDiarizer(config)
        d.diarize(emb_sess_test_dict=emb_sess_avg_dict, emb_seq_test= emb_scale_seq_dict, clus_test_label_dict=base_clus_label_dict)


        # create manifest file from rttm
        with open(pred_rttm, "r") as f:
            lines = f.readlines()
            
            # create AUDIO_RTTM_MAP
            AUDIO_RTTM_MAP = {}
            AUDIO_RTTM_MAP[lines[0].split()[1]] = {'audio_filepath': save_path,
            'rttm_filepath': pred_rttm,
            'offset': None,
            'duration': None,
            'text': None,
            'num_speakers': 2,
            'uem_filepath': None,
            'ctm_filepath': None}
            manifest_filepath = os.path.join(tmp_dir,get_uniqname_from_filepath(voice.filename) + "_manifest.json")
            write_rttm2manifest(AUDIO_RTTM_MAP, manifest_filepath)

            # with open(manifest_filepath,'a+') as fp:
            #     for idx, line in enumerate(lines):
            #         offset = float(line.split()[3])
            #         duration = float(line.split()[4])
            #         meta = {
            #             'audio_filepath': save_path, 
            #             'offset': offset, 
            #             'duration': duration, # write it manually 
            #             'label': 'U', 
            #             'uniq_id': idx
            #         } 
            #         json.dump(meta,fp)
            #         fp.write('\n')

        # create service
        # service = EmbeddingVoiceService()
        print("starting...")
        manifest_embs = b._extract_embeddings(manifest_filepath, 0, 1)
        # calculate result
        print("end")
        result = sound_similarity(manifest_embs, voice_emb)
        duration = []
        # print(f"result: {result}")
        for line, score in zip(lines, result):
            # print(f"score: {score}")
            if float(score) >= cfg.threshold:
                duration.append((float(line.split()[3]),float(line.split()[3])+ float(line.split()[4])))
        resp = {
            "speaker": name,
            "duration": duration
        }
    # shutil.rmtree(tmp_dir)
    return JSONResponse(content=resp)


async def diarize_audio(voice: UploadFile):
    if voice is None:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # create temporary store (remove after promgram finish)
    tmp_dir = "temporary_store"
    os.makedirs(tmp_dir, exist_ok=True)

    # write copy of input audio file
    content = await voice.read()
    save_path = os.path.join(tmp_dir, voice.filename)
    with open(save_path, 'wb+') as f:
        f.write(content)
    save_path = convert2wav(save_path)
    audio_processing(save_path)
    
    # # diarization
    # pred_rttm_path = speaker_diarization(audio_file=save_path)

    audio_dir = os.path.dirname(save_path)

    #create file input_manifest.json 
    meta = {
        'audio_filepath': save_path, #args.input_audio, 
        'offset': 0, 
        'duration':None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': 2, 
        'rttm_filepath': None,#phone_rttm, 
        'uem_filepath' : None
    }

    with open(os.path.join(audio_dir,'input_manifest.json'),'w+') as fp:
        json.dump(meta,fp)
        fp.write('\n')

    # create output folder (delete after promgram finish)
    output_dir = os.path.join(audio_dir, 'output')
    os.makedirs(output_dir,exist_ok=True)

    name = get_uniqname_from_filepath(save_path)
    pred_rttm = os.path.join(output_dir,f'pred_rttms/{name}.rttm')

    # add information for config file for clustering process
    config = OmegaConf.load(MODEL_CONFIG)

    config.diarizer.manifest_filepath = os.path.join(audio_dir, 'input_manifest.json')
    config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
    
    # data processsing
    audio_processing(save_path, factor=2)

    a = VAD(config)
    a._perform_speech_activity_detection()
    _speaker_manifest_path = a._speaker_manifest_path
    
    b = SpeakerEmbedding(config)
    embs_and_timestamps = b.diarizer(_speaker_manifest_path)

    c = ClusterEmbedding(config)
    emb_sess_avg_dict, emb_scale_seq_dict, base_clus_label_dict = c.prepare_cluster_embs_infer(embs_and_timestamps=embs_and_timestamps)

    d = NeuralDiarizer(config)
    d.diarize(emb_sess_test_dict=emb_sess_avg_dict, emb_seq_test= emb_scale_seq_dict, clus_test_label_dict=base_clus_label_dict)
    AUDIO_RTTM_MAP = {}
    # create manifest file from rttm
    with open(pred_rttm, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # create AUDIO_RTTM_MAP
            AUDIO_RTTM_MAP[line.split()[1] + f"_{idx}"] = {'audio_filepath': save_path,
            'rttm_filepath': pred_rttm,
            'offset': float(line.split()[3]),
            'duration': float(line.split()[4]),
            'text': None,
            'num_speakers': 2,
            'uem_filepath': None,
            'ctm_filepath': None}
            manifest_filepath = os.path.join(tmp_dir,get_uniqname_from_filepath(voice.filename) + "_manifest.json")
        write_rttm2manifest(AUDIO_RTTM_MAP, manifest_filepath)

        # with open(manifest_filepath,'a+') as fp:
        #     for idx, line in enumerate(lines):
        #         offset = float(line.split()[3])
        #         duration = float(line.split()[4])
        #         meta = {
        #             'audio_filepath': save_path, 
        #             'offset': offset, 
        #             'duration': duration, # write it manually 
        #             'label': 'U', 
        #             'uniq_id': idx
        #         } 
        #         json.dump(meta,fp)
        #         fp.write('\n')
    
    return manifest_filepath
    
@app.post("/diarization", tags=["voice"])
async def diarizer(voice: UploadFile, background_task: BackgroundTasks):
    background_task.add_task(diarize_audio, voice)
    return {'message': 'diarizing audio file...'}

    
@app.post("/recognize", tags=['voice'])
async def speaker_recognize(manifest_filepath: str, collection_name: Union[str, None]=None):
    db = Milvus(collection_name=collection_name)
    # output_fields = ["vector"]
    # res = db.col.query(expr=f"name like '{name}'", output_fields=output_fields)
    # if len(res) != 1:
    #     logging.error(f"There is no person or more than one person named {name}")
    #     resp = {
    #             "speaker": None,
    #             "duration": None
    #         }
    # else:
    # separate each 10 lines of manifest_filepath to a new file
    from utils import json_split
    print(f"manifest_filepath: {manifest_filepath}")
    list_of_manifest_file = json_split(manifest_filepath, 10)
    print(f"list_of_manifest_file: {list_of_manifest_file}")
    # duration = []
    # voice_emb = torch.as_tensor(np.array(res[0]['vector']))
    # load config
    cfg = OmegaConf.load(MODEL_CONFIG)
    cfg.diarizer.speaker_embeddings.parameters.save_embeddings = False
    # create service
    b = SpeakerEmbedding(cfg)
    resp = {}
    for file in list_of_manifest_file:
        tmp = []
        manifest_embs = b._extract_embeddings(file, 0, 1)
        # # calculate result
        # result = sound_similarity(manifest_embs, voice_emb)

        # with open(file, "r") as f:
        #     lines = f.readlines()
        #     for line, score in zip(lines, result):
        #         # print(f"score: {score}")
        #         if float(score) >= cfg.threshold:
        #             duration.append((float(line['offset']),float(line['offset'])+ float(line['duration'])))
        
        # search in vector database
        for _, value in manifest_embs.items():
            res = [db.search(value[i, :].tolist()) for i in range(value.shape[0])]
        print(res)
        if len(res) > 0 and all(len(v) > 0 for v in res):
            for idx, relevance_voice in enumerate(res):
                print(f"relevance_voice: {relevance_voice}")
                if relevance_voice[0][1] > cfg.threshold:
                    name = relevance_voice[0][0]['name']
                    tmp.append((name, idx))
            with open(file, 'r') as f:
                data = [json.loads(line) for line in f.readlines()]
                for pair in tmp:
                    if pair[0] in resp:
                        resp[pair[0]].append((data[pair[1]]['offset'], data[pair[1]]['offset'] + data[pair[1]]['duration']))
                    else:
                        resp[pair[0]] = [(data[pair[1]]['offset'], data[pair[1]]['offset'] + data[pair[1]]['duration'])]

        # os.remove(file)


        # resp = {
        #     "speaker": name,
        #     "duration": duration
        # }
        tmp_dir = Path(manifest_filepath).parent.absolute()
    # shutil.rmtree(tmp_dir)
    return JSONResponse(content=resp)
# @app.get("/verification", tags=["voice"])
# async def verification_only(rttm_file: str):

@app.delete("/refresh_db", tags=["voice"])
async def refresh_face_db(partition_name: str):
    # from pymilvus import utility
    try:
        utility.drop_collection(partition_name, using="default")
        directory = Path(Settings().GLOBAL_VOICE_DIR)/partition_name
        if directory.exists():
            shutil.rmtree(directory)
        return True
    except:
        raise HTTPException(status_code=400, detail="Cannot delete database")

@app.get("/list_collections")
async def list_col():
    cols = {"collections": utility.list_collections()}
    return JSONResponse(content=cols)

@app.get("/query")
async def query(collection_name: str, expr: str):
    output_fields = ["vector", "name"]
    a = Milvus(collection_name=collection_name)
    res = a.col.query(expr=expr, output_fields=output_fields)
    print(res)
    return None

@app.get("/search")
async def search(collection_name: str, embeddings: list):
    a= Milvus(collection_name=collection_name)
    res = a.search(embedding=embeddings)
    res = {"result": res}
    return JSONResponse(content=res)

