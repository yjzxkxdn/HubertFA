import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils

from torchaudio.transforms import Resample
from transformers import Wav2Vec2FeatureExtractor, HubertModel

from networks.hubert.model import HubertSoft
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


class Audio2HubertSoft(torch.nn.Module):
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320):
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
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320):
        super().__init__()
        print(' [Encoder Model] Chinese Hubert')
        print(' [Loading] ' + path)
        self.model = HubertModel.from_pretrained(path, local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            path, local_files_only=True)

    def forward(self,
                audio):  # B, T
        with torch.inference_mode():
            input_values = self.feature_extractor(audio, return_tensors="pt",
                                                  sampling_rate=16000).input_values.to(audio.device).squeeze(1)
            return self.model(input_values)["last_hidden_state"]  # [1, T, C]


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


class Audio2ContentVec768L12TTA2X:
    def __init__(self, path, h_sample_rate=16000, h_hop_size=160, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 12,  # layer 12
        }
        with torch.no_grad():
            feats = self.hubert.extract_features(**inputs)[0]
            inputs["source"] = F.pad(inputs["source"], (160, 0))
            feats2 = self.hubert.extract_features(**inputs)[0]
            n = feats2.shape[1] - feats.shape[1]
            if n > 0:
                feats = F.pad(feats, (0, 0, 0, 1))
            feats_tta = torch.cat((feats2, feats), dim=2).reshape(feats.shape[0], -1, feats.shape[-1])
            feats_tta = feats_tta[:, 1:, :]
            if n > 0:
                feats_tta = feats_tta[:, :-1, :]
        units = feats_tta  # .transpose(2, 1)
        return units  # [1, T, B]


class UnitsEncoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate=16000, encoder_hop_size=320, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        is_loaded_encoder = False
        if encoder == 'hubertsoft':
            self.model = Audio2HubertSoft(encoder_ckpt).to(device)
            is_loaded_encoder = True
        if encoder == 'cnhubert':
            self.model = Audio2CNHubert(encoder_ckpt).to(device)
            is_loaded_encoder = True
        if encoder == 'hubertsofttta2x':
            self.model = Audio2HubertSoftTTA2X(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec768l12tta2x':
            self.model = Audio2ContentVec768L12TTA2X(encoder_ckpt, device=device)
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
        return units_aligned
