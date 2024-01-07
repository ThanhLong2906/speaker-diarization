from omegaconf import OmegaConf, DictConfig
import logging
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from utils import get_available_model_names
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
)
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    # get_embs_and_timestamps,
    get_uniqname_from_filepath,
    # parse_scale_configs,
    # perform_clustering,
    # segments_manifest_to_subsegments_manifest,
    validate_vad_manifest,
    write_rttm2manifest,
)
import os
import shutil
import json
import torch
from tqdm import tqdm
from copy import deepcopy
try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


class VAD:
    def __init__(self, config) -> None:
        self._cfg = config
        self._diarizer_params = self._cfg.diarizer
        self._vad_params = self._cfg.diarizer.vad.parameters
        self._out_dir = self._cfg.diarizer.out_dir
        # self.out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(self._out_dir, exist_ok=True)
        self._vad_dir = os.path.join(self._out_dir, 'vad_outputs')
        self._vad_out_file = os.path.join(self._vad_dir, "vad_out.json")
        self._init_vad_model()
        self.AUDIO_RTTM_MAP = audio_rttm_map(self._diarizer_params.manifest_filepath)

    def _init_vad_model(self):
        """
        Initialize VAD model with model name or path passed through config
        """
        model_path = self._cfg.diarizer.vad.model_path
        if model_path.endswith('.nemo'):
            self._vad_model = EncDecClassificationModel.restore_from(model_path, map_location=self._cfg.device)
            logging.info("VAD model loaded locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(EncDecClassificationModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "vad_telephony_marblenet"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self._vad_model = EncDecClassificationModel.from_pretrained(
                model_name=model_path, map_location=self._cfg.device
            )
        self._vad_window_length_in_sec = self._vad_params.window_length_in_sec
        self._vad_shift_length_in_sec = self._vad_params.shift_length_in_sec
        self.has_vad_model = True
    
    def _setup_vad_test_data(self, manifest_vad_input):
        vad_dl_config = {
            'manifest_filepath': manifest_vad_input,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': self._cfg.get('batch_size'),
            'vad_stream': True,
            'labels': ['infer',],
            'window_length_in_sec': self._vad_window_length_in_sec,
            'shift_length_in_sec': self._vad_shift_length_in_sec,
            'trim_silence': False,
            'num_workers': self._cfg.num_workers,
        }
        self._vad_model.setup_test_data(test_data_config=vad_dl_config)
    
    def _run_vad(self, manifest_file):
        """
        Run voice activity detection. 
        Get log probability of voice activity detection and smoothes using the post processing parameters. 
        Using generated frame level predictions generated manifest file for later speaker embedding extraction.
        input:
        manifest_file (str) : Manifest file containing path to audio file and label as infer

        """

        shutil.rmtree(self._vad_dir, ignore_errors=True)
        os.makedirs(self._vad_dir)

        self._vad_model.eval()

        time_unit = int(self._vad_window_length_in_sec / self._vad_shift_length_in_sec)
        trunc = int(time_unit / 2)
        trunc_l = time_unit - trunc
        all_len = 0
        data = []
        for line in open(manifest_file, 'r', encoding='utf-8'):
            file = json.loads(line)['audio_filepath']
            data.append(get_uniqname_from_filepath(file))

        status = get_vad_stream_status(data)
        for i, test_batch in enumerate(
            tqdm(self._vad_model.test_dataloader(), desc='vad', leave=True, disable=not self.verbose)
        ):
            test_batch = [x.to(self._vad_model.device) for x in test_batch]
            with autocast():
                log_probs = self._vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
                probs = torch.softmax(log_probs, dim=-1)
                pred = probs[:, 1]
                if status[i] == 'start':
                    to_save = pred[:-trunc]
                elif status[i] == 'next':
                    to_save = pred[trunc:-trunc_l]
                elif status[i] == 'end':
                    to_save = pred[trunc_l:]
                else:
                    to_save = pred
                all_len += len(to_save)
                outpath = os.path.join(self._vad_dir, data[i] + ".frame")
                with open(outpath, "a", encoding='utf-8') as fout:
                    for f in range(len(to_save)):
                        fout.write('{0:0.4f}\n'.format(to_save[f]))
            del test_batch
            if status[i] == 'end' or status[i] == 'single':
                all_len = 0

        if not self._vad_params.smoothing:
            # Shift the window by 10ms to generate the frame and use the prediction of the window to represent the label for the frame;
            self.vad_pred_dir = self._vad_dir
            frame_length_in_sec = self._vad_shift_length_in_sec
        else:
            # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
            # smoothing_method would be either in majority vote (median) or average (mean)
            logging.info("Generating predictions with overlapping input segments")
            smoothing_pred_dir = generate_overlap_vad_seq(
                frame_pred_dir=self._vad_dir,
                smoothing_method=self._vad_params.smoothing,
                overlap=self._vad_params.overlap,
                window_length_in_sec=self._vad_window_length_in_sec,
                shift_length_in_sec=self._vad_shift_length_in_sec,
                num_workers=self._cfg.num_workers,
            )
            self.vad_pred_dir = smoothing_pred_dir
            frame_length_in_sec = 0.01

        logging.info("Converting frame level prediction to speech/no-speech segment in start and end times format.")

        vad_params = self._vad_params if isinstance(self._vad_params, (DictConfig, dict)) else self._vad_params.dict()
        table_out_dir = generate_vad_segment_table(
            vad_pred_dir=self.vad_pred_dir,
            postprocessing_params=vad_params,
            frame_length_in_sec=frame_length_in_sec,
            num_workers=self._cfg.num_workers,
            out_dir=self._vad_dir,
        )

        AUDIO_VAD_RTTM_MAP = {}
        for key in self.AUDIO_RTTM_MAP:
            if os.path.exists(os.path.join(table_out_dir, key + ".txt")):
                AUDIO_VAD_RTTM_MAP[key] = deepcopy(self.AUDIO_RTTM_MAP[key])
                AUDIO_VAD_RTTM_MAP[key]['rttm_filepath'] = os.path.join(table_out_dir, key + ".txt")
            else:
                logging.warning(f"no vad file found for {key} due to zero or negative duration")

        write_rttm2manifest(AUDIO_VAD_RTTM_MAP, self._vad_out_file)
        self._speaker_manifest_path = self._vad_out_file

    def _perform_speech_activity_detection(self):
        """
        Checks for type of speech activity detection from config. Choices are NeMo VAD,
        external vad manifest and oracle VAD (generates speech activity labels from provided RTTM files)
        """
        if self.has_vad_model:
            self._auto_split = True
            self._split_duration = self._diarizer_params.vad.split_duration #10
            manifest_vad_input = self._diarizer_params.manifest_filepath

            if self._auto_split:
                logging.info("Split long audio file to avoid CUDA memory issue")
                logging.debug("Try smaller split_duration if you still have CUDA memory issue")
                config = {
                    'input': manifest_vad_input,
                    'window_length_in_sec': self._vad_window_length_in_sec,
                    'split_duration': self._split_duration,
                    'num_workers': self._cfg.num_workers,
                    'out_dir': self._diarizer_params.out_dir,
                }
                manifest_vad_input = prepare_manifest(config)
            else:
                logging.warning(
                    "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
                )

            self._setup_vad_test_data(manifest_vad_input)
            self._run_vad(manifest_vad_input)

        elif self._diarizer_params.vad.external_vad_manifest is not None:
            self._speaker_manifest_path = self._diarizer_params.vad.external_vad_manifest
        elif self._diarizer_params.oracle_vad:
            self._speaker_manifest_path = os.path.join(self._speaker_dir, 'oracle_vad_manifest.json')
            self._speaker_manifest_path = write_rttm2manifest(self.AUDIO_RTTM_MAP, self._speaker_manifest_path)
        else:
            raise ValueError(
                "Only one of diarizer.oracle_vad, vad.model_path or vad.external_vad_manifest must be passed from config"
            )
        # validate_vad_manifest(self.AUDIO_RTTM_MAP, vad_manifest=self._speaker_manifest_path)

    @property
    def verbose(self) -> bool:
        return self._cfg.verbose

def main():
    input_config = "../../input_config/diar_infer_telephonic.yaml"
    cfg = OmegaConf.load(input_config)
    a = VAD(cfg)
    
    a._perform_speech_activity_detection()
    print(a._speaker_manifest_path)
if __name__=='__main__':
    main()