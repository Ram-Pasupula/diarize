from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
from process import transcriber
from fastapi.responses import StreamingResponse
import time
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
LANGUAGE_CODES = sorted(("en", "es"))


app = FastAPI(
    title="Whisper API",
    debug=False,

    docs_url="/app/v1",
    redoc_url="/api/redoc",
    generate_schema=False,
)
  
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def transcribe_and_stream(file, task, lang, output):
    chunks = transcriber(file, task, lang, output)
    return chunks


@app.post("/transcode")
async def asr(file: UploadFile = File(...),
              task: Union[str, None] = Query(default="transcribe", enum=[
                  "transcribe"]),
              lang: Union[str, None] = Query(
                  default="en", enum=LANGUAGE_CODES),
              output: Union[str, None] = Query(
    default="txt", enum=["txt",  "json"])
):

    try:
        start_time = time.time()
        audio_content = await file.read()
        logger.info(f"task : {task}")
        logger.info(f"lang : {lang}")
        response = await transcribe_and_stream(audio_content, task, lang, output)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time} seconds")
        logger.info(f"Execution time: {elapsed_time} seconds")
        return response
    except Exception as e:
        logger.error(f'Error during transcription:{e}')
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)