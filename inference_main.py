import io
import logging
import time
from pathlib import Path
from IPython.display import Audio
import numpy as np
import soundfile
import argparse

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
parser = argparse.ArgumentParser(description='sovits inference')
parser.add_argument('-m', '--model_path', type=str, default="models/G_0.pth", help='模型路径')
parser.add_argument('-c', '--config_path', type=str, default="models/config.json", help='配置文件路径')
parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["君の知らない物語-src.wav"], help='wav文件名列表，放在raw文件夹下')
parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='音高调整，支持正负（半音）')
parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['nen'], help='合成目标说话人名称')
args = parser.parse_args()

svc_model = Svc(args.model_path, args.config_path)
infer_tool.mkdir(["raw", "results"])

clean_names = args.clean_names
trans = args.trans
spk_list = args.spk_list
slice_db = args.slice_db
wav_format = args.wav_format

infer_tool.fill_a_to_b(trans, clean_names)
for clean_name, tran in zip(clean_names, trans):
    raw_audio_path = f"raw/{clean_name}"
    if "." not in raw_audio_path:
        raw_audio_path += ".wav"
    infer_tool.format_wav(raw_audio_path)
    wav_path = Path(raw_audio_path).with_suffix('.wav')
    chunks = slicer.cut(wav_path, db_thresh=slice_db)
    audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

    for spk in spk_list:
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
            raw_path = io.BytesIO()
            soundfile.write(raw_path, data, audio_sr, format="wav")
            raw_path.seek(0)
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
            else:
                out_audio, out_sr = svc_model.infer(spk, tran, raw_path)
                _audio = out_audio.cpu().numpy()
            audio.extend(list(_audio))

        res_path = f'./results/{clean_name}_{tran}key_{spk}.{wav_format}'
        soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
Audio(f'./results/{clean_name}_{tran}key_{spk}.{wav_format}')
