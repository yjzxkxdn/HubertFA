import os.path
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from networks.hubert.hubert_model import hubert_soft, get_units
from networks.hubert.vec_model import load_model, get_vec_units


class HubertEncoder:
    def __init__(self, pt_type='hubert', pt_path='checkpoints/hubert/hubert_soft.pt', use_gpu=False):
        self.pt_type = pt_type
        if self.pt_type == 'vec':
            pt_path = "checkpoints/vec/checkpoint_best_legacy_500.pt"
            self.dev = torch.device("cuda")
            self.hbt_model = load_model(pt_path)
        else:
            pt_path = list(Path(pt_path).parent.rglob('*.pt'))[0]
            self.dev = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            self.hbt_model = hubert_soft(str(pt_path)).to(self.dev)

    def encode(self, wav_path):
        if isinstance(wav_path, BytesIO):
            npy_path = ""
            wav_path.seek(0)
        else:
            npy_path = Path(wav_path).with_suffix('.npy')
        if os.path.exists(npy_path):
            units = np.load(str(npy_path))
        elif self.pt_type == 'vec':
            units = get_vec_units(self.hbt_model, wav_path, self.dev).cpu().numpy()[0]
        else:
            units = get_units(self.hbt_model, wav_path, self.dev).cpu().numpy()[0]
        return units  # [T,256]
