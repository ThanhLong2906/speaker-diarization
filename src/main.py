from fastapi import FastAPI, HTTPException, UploadFile, Body
# import shortuu
# from utils import TimeCounter
import os
# import time
from contextlib import asynccontextmanager
# from contracts import AddFaceRequest
from services.embedding_service import EmbeddingVoiceService, sound_similarity
from services.speaker_diarization_service import speaker_diarization
# from services.extract_video_service import ExtractVideoService
from pymilvus import connections, utility
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
from services.speaker_diarization_service import audio_processing
from nemo_from_scratch.speaker_embedding import SpeakerEmbedding
import json

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
async def add_voice(voice: UploadFile, name: str, collection_name: Union[str, None]=None):
    if voice is None:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # task = TimeCounter(task_name="Saving received image to server")
    content = await voice.read()
    dot_index = voice.filename.rfind(".")
    # file_name = people_metadatas.people_metadatas[0].id + "-" + str(round(time.time())) + img.filename[dot_index:]
    file_name = name + voice.filename[dot_index:]
    os.makedirs("../voice_db",exist_ok=True)
    save_path = f"../voice_db/{file_name}"
    with open(save_path, 'wb+') as f:
        f.write(content)
    audio_processing(save_path)
    # task.done()
    # laod config file
    cfg = OmegaConf.load(MODEL_CONFIG)
    # task = TimeCounter(task_name="Extracting faces and save to database")
    service = SpeakerEmbedding(cfg)
    metadata = [{"name":name}]
    service.save_voice(save_path, metadata, collection_name=collection_name)
    # task.done()

    return {"id": file_name}

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


@app.post("/diarization", tags=["voice"])
async def diarization(voice: UploadFile):
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

    del(a._vad_model)
    del(b._speaker_model)
    del(d.msdd_model)

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
    return manifest_filepath
    
@app.post("/recognize", tags=['voice'])
async def speaker_recognize(manifest_filepath: str, name: str, collection_name: Union[str, None]=None):
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
            # load config
            cfg = OmegaConf.load(MODEL_CONFIG)
            # create service
            b = SpeakerEmbedding(cfg)
            print("starting...")
            manifest_embs = b._extract_embeddings(manifest_filepath, 0, 1)
            print("end")

            # query

            # calculate result
            result = sound_similarity(manifest_embs, voice_emb)
            duration = []

            with open(manifest_filepath, "r") as f:
                lines = f.readlines()
                for line, score in zip(lines, result):
                    # print(f"score: {score}")
                    if float(score) >= cfg.threshold:
                        duration.append((float(line['offset']),float(line['offset'])+ float(line['duration'])))
                resp = {
                    "speaker": name,
                    "duration": duration
                }
            # shutil.rmtree(tmp_dir)
        return JSONResponse(content=resp)
# @app.get("/verification", tags=["voice"])
# async def verification_only(rttm_file: str):


@app.delete("/refresh_db", tags=["voice"])
async def refresh_face_db(partition_name: str):
    # from pymilvus import utility
    try:
        utility.drop_collection(partition_name, using="default")
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

