# import os
import json
# import logging
from nemo.collections.asr.parts.utils.speaker_utils import get_uniqname_from_filepath
from nemo.utils import logging

def get_available_model_names(class_name):
    "lists available pretrained model names from NGC"
    available_models = class_name.list_available_models()
    return list(map(lambda x: x.pretrained_model_name, available_models))

# def get_uniq_id_with_dur(meta, decimals=3):
#     """
#     Return basename with offset and end time labels
#     """
#     bare_uniq_id = get_uniqname_from_filepath(meta['audio_filepath'])
#     # bare_uniq_id = get_uniqname_from_filepath(meta['rttm_filepath'])
#     if meta['offset'] is None and meta['duration'] is None:
#         return bare_uniq_id
#     if meta['offset']:
#         offset = str(int(round(meta['offset'], decimals) * pow(10, decimals)))
#     else:
#         offset = 0
#     if meta['duration']:
#         endtime = str(int(round(meta['offset'] + meta['duration'], decimals) * pow(10, decimals)))
#     else:
#         endtime = 'NULL'
#     uniq_id = f"{bare_uniq_id}_{offset}_{endtime}"
#     return uniq_id

# def audio_rttm_map(manifest, attach_dur=False):
#     """
#     This function creates AUDIO_RTTM_MAP which is used by all diarization components to extract embeddings,
#     cluster and unify time stamps
#     Args: manifest file that contains keys audio_filepath, rttm_filepath if exists, text, num_speakers if known and uem_filepath if exists

#     returns:
#     AUDIO_RTTM_MAP (dict) : A dictionary with keys of uniq id, which is being used to map audio files and corresponding rttm files
#     """

#     AUDIO_RTTM_MAP = {}
#     with open(manifest, 'r') as inp_file:
#         lines = inp_file.readlines()
#         logging.info("Number of files to diarize: {}".format(len(lines)))
#         for line in lines:
#             line = line.strip()
#             dic = json.loads(line)

#             meta = {
#                 'audio_filepath': dic['audio_filepath'],
#                 'rttm_filepath': dic.get('rttm_filepath', None),
#                 'offset': dic.get('offset', None),
#                 'duration': dic.get('duration', None),
#                 'text': dic.get('text', None),
#                 'num_speakers': dic.get('num_speakers', None),
#                 'uem_filepath': dic.get('uem_filepath', None),
#                 'ctm_filepath': dic.get('ctm_filepath', None),
#             }
#             if attach_dur:
#                 uniqname = get_uniq_id_with_dur(meta)
#             else:
#                 uniqname = get_uniqname_from_filepath(filepath=meta['audio_filepath'])

#             if uniqname not in AUDIO_RTTM_MAP:
#                 AUDIO_RTTM_MAP[uniqname] = meta
#             else:
#                 raise KeyError(
#                     "file {} is already part of AUDIO_RTTM_MAP, it might be duplicated, Note: file basename must be unique".format(
#                         meta['audio_filepath']
#                     )
#                 )

#     return AUDIO_RTTM_MAP