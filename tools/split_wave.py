from pathlib import Path
import torch
import yaml
from torchaudio.transforms import Resample

from networks.hnsep.nets import CascadedNet

class SplitWave:
    def __init__(self, hnspe_model_path, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        config_file = Path(hnspe_model_path).parent / 'config.yaml'
        with open(config_file, "r") as config:
            args = yaml.safe_load(config)
        self.args = args
        model = CascadedNet(
                    args["n_fft"], 
                    args["hop_length"], 
                    args["n_out"], 
                    args["n_out_lstm"], 
                    True, 
                    is_mono=args["is_mono"],
                    fixed_length = True)
        model.to(device)
        print("[SplitWave Model]")
        print("[Loading]" + hnspe_model_path)
        model.load_state_dict(torch.load(hnspe_model_path, map_location='cpu'))
        model.eval()
        self.model = model
        self.resample_kernel = {}
        
    def split_from_audio(
        self,
        audio,  # B, T
        sample_rate,
    ):
        # resample
        if sample_rate == self.args["sr"]:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.args["sr"],
                                                         lowpass_filter_width=128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        
        # split
        with torch.no_grad():
            audio_res = audio_res.to(self.device)
            audio_harmonic = self.model.predict_fromaudio(audio_res)
            audio_noise = audio_res - audio_harmonic
            
        return audio_harmonic, audio_noise
            
            