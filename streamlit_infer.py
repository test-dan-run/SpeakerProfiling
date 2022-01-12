import os
import torch
import torchaudio
import sounddevice as sd
import streamlit as st
import soundfile as sf
from NISP.lightning_model import LightningModel
from config import TestNISPConfig as cfg
 
def record(length, sample_rate, channels, output_path):

    stream = sd.rec(int(length * sample_rate), samplerate=sample_rate, channels=channels)
    print('Recording audio for 10s...')

    sd.wait()
    sf.write(output_path, stream, sample_rate)

    print('Recording complete. Audio can be found at {}'.format(output_path))

def predict(audio_path, model_ckpt, csv_path, input_length, slice_window, sample_rate):

    # Load audio as torch array
    arr, _ = torchaudio.load(audio_path)
    # Split input audio into slices of input_length seconds
    slices = arr.unfold(1, input_length*sample_rate, slice_window*sample_rate).transpose(0,1)

    # load model checkpoint
    model = LightningModel.load_from_checkpoint(model_ckpt, csv_path=csv_path)
    model.eval()

    # predict
    predictions = []
    with torch.no_grad():
        for slice in slices:
            h_pred, _, _ = model(slice)
            predictions.append((h_pred.view(-1) * model.h_std + model.h_mean).item())

    mean = sum(predictions)/len(predictions)
    print('Mean Predicted Height:', mean)
    with open('output.txt', 'w') as f:
        f.write(str(round(mean, 2)))

def show_audio_widget(path):
    audio_file = open(path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

if __name__ == '__main__':

    st.title('Height Recogniser v0.0.1')

    record_button = st.button(
        'Record', on_click=record, args=(cfg.record_seconds, cfg.sample_rate, cfg.channels, cfg.wav_output_filename))

    if (not record_button and os.path.isfile(cfg.wav_output_filename)) or record_button:
        show_audio_widget(cfg.wav_output_filename)

    predict_button = st.button(
        'Predict', on_click=predict, args=(cfg.wav_output_filename, cfg.model_checkpoint, cfg.csv_path, cfg.slice_seconds, cfg.slice_window, cfg.sample_rate))
    
    if predict_button:
        with open('output.txt', 'r') as f:
            lines = f.readlines()
            txt = lines[0]
        st.write('Your predicted height is {}'.format(txt))
