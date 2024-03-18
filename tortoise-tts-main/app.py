import streamlit as st
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from glob import glob
import io
import librosa 
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.io.wavfile import read

# def load_wav_to_torch(full_path):
#     sampling_rate, data = read(full_path)
#     if data.dtype == np.int32:
#         norm_fix = 2 ** 31
#     elif data.dtype == np.int16:
#         norm_fix = 2 ** 15
#     elif data.dtype == np.float16 or data.dtype == np.float32:
#         norm_fix = 1.
#     else:
#         raise NotImplemented(f"Provided data dtype not supported: {data.dtype}")
#     return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)


def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str):
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        # elif audiopath.endswith('.wav'):
        #     audio, lsr = load_wav_to_torch(audiopath)
        else:
            assert False, F"Ubsupported audio format provided: {audiopath[-4:]}"
    
    elif isinstance(audiopath, io.BytesIO):
        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0]
    
    # # Remove any channel data.
    # if len(audio.shape) > 1:
    #     if audio.shape[0] < 5:
    #         audio = audio[0]
    #     else:
    #         assert audio.shape[1] < 5
    #         audio = audio[:, 0]

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max={audio.max()} Min={audio.min()}.")
    
    audio.clip_(-1,1)

    return audio.unsqueeze(0)


def classify_audio_clip(clip):
    """
    Returns whether or not Tortoises' classifier thinks the given clip came from Tortoise.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: True if the clip was classified as coming from Tortoise and false if it was classified as real.
    """
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict =  torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

st.set_page_config(layout="wide")

def main():
    st.title("Check Voice Print")

    uploadFile = st.file_uploader("Please select The file" , type=["mp3","wav"])

    if uploadFile is not None:
        if st.button("   Ckeck   "):
            col1, col2 = st.columns(2)

            with col1:
                st.info("Result")

                audio_clip = load_audio(uploadFile)
                result = classify_audio_clip(audio_clip)
                result = result.item()
                st.info(f"Result probability : { result : .2f}")
                st.success(f"The uploaded audio is { result * 100 : .2f}% likly to be AI generated")

            with col2:
                st.info("you uploaded audio is below")
                st.audio(uploadFile)

                fig = px.line()
                fig.add_scatter(x= list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
                fig.update_layout(
                    title = "Plot",
                    xaxis_title = "Time",
                    yaxis_title= "Amplitude"
                )

                st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    main()