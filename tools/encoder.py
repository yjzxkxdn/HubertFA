import torch
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor, HubertModel

from networks.hubert.model import HubertSoft


class UnitsEncoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate=16000, encoder_hop_size=320, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        is_loaded_encoder = False
        if encoder == 'hubertsoft':
            self.model = Audio2HubertSoft(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'cnhubert':
            self.model = Audio2CNHubert(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'whisper-ppg':
            self.model = Audio2Whisper(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertsofttta2x':
            self.model = Audio2HubertSoftTTA2X(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if not is_loaded_encoder:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")

        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size

    def encode(self,
               audio,  # B, T
               sample_rate,
               hop_size):
        # resample
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate,
                                                         lowpass_filter_width=128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)

        # encode
        if audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res)

        # alignment
        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max=units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
        return units_aligned.transpose(1, 2)  # [B, C, T]


class Audio2HubertSoft(torch.nn.Module):
    def __init__(self, path, device='cpu', h_sample_rate=16000, h_hop_size=320):
        super().__init__()
        print(' [Encoder Model] HuBERT Soft')
        self.hubert = HubertSoft()
        print(' [Loading] ' + path)
        checkpoint = torch.load(path)["hubert"]
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.hubert.eval()

    def forward(self,
                audio):  # B, T
        with torch.inference_mode():
            units = self.hubert.units(audio.unsqueeze(1))
            return units  # [1, T, C]


class Audio2CNHubert(torch.nn.Module):
    def __init__(self, path, device='cpu', h_sample_rate=16000, h_hop_size=320):
        super().__init__()
        print(' [Encoder Model] Chinese Hubert')
        print(' [Loading] ' + path)
        self.model = HubertModel.from_pretrained(path, local_files_only=True).to(device)
        self.model.eval()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            path, local_files_only=True)

    def forward(self,
                audio):  # B, T
        with torch.inference_mode():
            input_values = self.feature_extractor(audio, return_tensors="pt",
                                                  sampling_rate=16000).input_values.to(audio.device).squeeze(1)
            return self.model(input_values)["last_hidden_state"]  # [1, T, C]


class Audio2Whisper(torch.nn.Module):
    def __init__(self, path, device='cpu', h_sample_rate=16000, h_hop_size=320):
        super().__init__()
        print(' [Encoder Model] Whisper')
        print(' [Loading] ' + path)\
            
        from whisper.audio import log_mel_spectrogram, pad_or_trim
        from whisper.model import ModelDimensions, Whisper

        self.dev = device
        self.pad_or_trim = pad_or_trim
        self.log_mel_spectrogram = log_mel_spectrogram
        checkpoint = torch.load(path, map_location=self.dev)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.hidden_dim = dims
        self.model = model.to(self.dev)

    def forward(self,
                audio):  # B, T
        audln = audio.shape[1]
        ppgln = audln // 320
        audio = self.pad_or_trim(audio)
        mel = self.log_mel_spectrogram(audio).to(self.dev)
        with torch.no_grad():
            ppg = self.model.encoder(mel).squeeze().data.cpu().float().numpy()
            ppg = torch.FloatTensor(ppg[:ppgln, ]).to(self.dev)
            return ppg[None, :, :]  # [1, T, C]


class Audio2HubertSoftTTA2X:
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Hubert Soft with TTA 2X')
        print(' [Loading] ' + path)
        self.hubert = HubertSoft()
        checkpoint = torch.load(path)["hubert"]
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.hubert = self.hubert.to(device)
        self.hubert.eval()

    def __call__(self, audio):
        # audio: [B, T]
        with torch.no_grad():
            feats = self.hubert.units(audio.unsqueeze(1))
            padded_audio = F.pad(audio, (160, 0))  # [B, T + pad_amount]
            feats2 = self.hubert.units(padded_audio.unsqueeze(1))
            n = feats2.shape[1] - feats.shape[1]
            if n > 0:
                feats = F.pad(feats, (0, 0, 0, 1))
            feats_tta = torch.cat((feats2, feats), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
            feats_tta = feats_tta[:, 1:, :]
            if n > 0:
                feats_tta = feats_tta[:, :-1, :]
        units = feats_tta  # .transpose(2, 1)
        return units  # [1, T, B]
