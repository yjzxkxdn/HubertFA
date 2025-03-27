import torchaudio


def load_wav(path, device, sample_rate=None):
    waveform, sr = torchaudio.load(str(path))
    if sample_rate != sr and sample_rate is not None:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform[0].to(device)
