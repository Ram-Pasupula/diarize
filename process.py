
import json
from io import StringIO
from threading import Lock
import logging
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

MODEL_PATH_W = "/Users/mac/Downloads/whisper-large"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH_W, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH_W)

whisper = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
    batch_size=24
)
model_lock = Lock()


def transcriber(
        audio,
        task,
        lang,
        output
):
    logging.info("trnascriber started")
    with model_lock:
        kwargs = {"language": f"{lang}", "task": f"{task}"}
        logger.info(f"generate_kwargs :{kwargs}")

        result = whisper(audio,
                         generate_kwargs=kwargs,
                         return_timestamps=True,
                         )
    return result
