import gradio as gr
import numpy as np
import librosa
import torch

from NISP.lightning_model import LightningModel
from config import TestNISPConfig as cfg

# load model checkpoint
model = LightningModel.load_from_checkpoint(cfg.model_checkpoint, csv_path=cfg.csv_path)
model.eval()

def predict_height(audio):

    # resample audio to required format (16kHz Sample Rate, Mono)
    input_sr, arr = audio
    arr = arr.astype(np.float32, order='C') / 32768.0
    arr = librosa.to_mono(arr.T)
    arr = librosa.resample(arr, input_sr, cfg.sample_rate)

    # convert to torch tensor
    tensor = torch.Tensor([arr])

    sample_length = cfg.slice_seconds * cfg.sample_rate
    win_length = cfg.slice_window * cfg.sample_rate

    if tensor.shape[-1] < sample_length:
        tensor = torch.nn.functional.pad(tensor, (0, sample_length - tensor.size(1)), 'constant')
        slices = tensor.unsqueeze(dim=0)
    else:
        # Split input audio into slices of input_length seconds
        slices = tensor.unfold(1, sample_length, win_length).transpose(0,1)

    # predict
    predictions = []
    with torch.no_grad():
        for slice in slices:
            h_pred, _, _ = model(slice)
            predictions.append((h_pred.view(-1) * model.h_std + model.h_mean).item())

    mean = sum(predictions)/len(predictions)

    return 'Your height is {}!'.format(round(mean, 2))

iface = gr.Interface(
  fn=predict_height,
  inputs='mic',
  outputs='text'
).launch()