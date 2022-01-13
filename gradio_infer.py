import gradio as gr
import numpy as np
import librosa
import torch

from NISP.lightning_model import LightningModel
from config import TestNISPConfig as cfg

INFERENCE_COUNT = 0

# load model checkpoint
model = LightningModel.load_from_checkpoint(cfg.model_checkpoint, csv_path=cfg.csv_path)
model.eval()

def predict_height(audio):

    global INFERENCE_COUNT
    INFERENCE_COUNT += 1

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
    h_preds, a_preds, g_preds = [], [], []
    with torch.no_grad():
        for slice in slices:
            h_pred, a_pred, g_pred = model(slice)
            h_preds.append((h_pred.view(-1) * model.h_std + model.h_mean).item())
            a_preds.append((a_pred.view(-1) * model.a_std + model.a_mean).item())
            g_preds.append(g_pred.view(-1).item())
    
    height = round(sum(h_preds)/len(h_preds),2)
    age = int(sum(a_preds)/len(a_preds))
    gender = 'Female' if sum(g_preds)/len(g_preds) > 0.5 else 'Male'

    print('Inference was run. Current inference count:', INFERENCE_COUNT)

    return 'You\'re {}, your height is {}, and you are {} years old.'.format(gender, height, age)

iface = gr.Interface(
  fn=predict_height,
  inputs='mic',
  outputs='text',
  description='Predicts your height, age, gender based on your voice. \n Ideally, a clip of more than 5 seconds would be preferred. Any less, and your clip will be zero-padded to 5 seconds.'
).launch(share=True)