import json
from io import StringIO
from threading import Lock
import logging
import torchaudio
# pip install pyannote.audio
from pyannote.audio import Pipeline
import requests

from typing import List, Optional, Union

import numpy as np
import requests
import torch
from pyannote.audio import Pipeline
from torchaudio import functional as F
from diarize_process import DiarizationPipeline
from transformers.pipelines.audio_utils import ffmpeg_read

# Replace with your actual backend API URL
BACKEND_API_URL = "http://127.0.0.1:8000/transcode"


def transcribe(audio_data, lang="en"):

    audio_file_name = audio_data.split("/")[-1]

    # Create a tuple for file upload
    files = {'file': (audio_file_name, open(audio_data, 'rb'))}
    params = {"task": "transcribe", "lang": lang, "output": "txt"}
    response = requests.post(BACKEND_API_URL, files=files, params=params)
    return response.text


def process(transcript, new_segments):
    group_by_speaker = True
    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
    segmented_preds = []
    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        if end_timestamps.any():
            upto_idx = np.argmin(np.abs(end_timestamps - end_time))
            if group_by_speaker:
                segmented_preds.append(
                    {
                        "speaker": segment["speaker"],
                        "text": "".join([chunk["text"] for chunk in transcript[: upto_idx + 1]]),
                        "timestamp": (transcript[0]["timestamp"][0], transcript[upto_idx]["timestamp"][1]),
                    }
                )
            else:
                for i in range(upto_idx + 1):
                    segmented_preds.append(
                        {"speaker": segment["speaker"], **transcript[i]})

                transcript = transcript[upto_idx + 1:]
                end_timestamps = end_timestamps[upto_idx + 1:]

    return segmented_preds


diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", cache_dir="./speaker-diarization"
)

pipeline = DiarizationPipeline(
    diarization_pipeline=diarization_pipeline
)
# result = diarization_pipeline("/Users/mac/Downloads/source/005-Greetings.mp3")
segments = pipeline("/Users/mac/Downloads/chunks/output_001.wav")

Transcribe = transcribe("/Users/mac/Downloads/chunks/output_001.wav")
tra = json.loads(Transcribe)
transcript = tra['chunks']
print(transcript)
res = process(transcript, segments)
print(res)

