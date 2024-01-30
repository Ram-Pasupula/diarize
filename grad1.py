import gradio as gr
import torch
from transformers import pipeline

# Change to "your-username/the-name-you-picked"
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

MODEL_PATH_W = "/Users/mac/Downloads/whisper-large"
transcribe_txt = " "
whisper = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH_W,
    device=device,
    torch_dtype=torch_dtype
)


def transcribe(audio_file, lang):
    global transcribe_txt
    transcribe_txt = whisper(audio_file)["text"]
    return transcribe_txt


def summarization_function():
    summarized_text = f" {transcribe_txt}"
    return summarized_text


def analysis_function():
    analysis_result = f" {transcribe_txt}"
    return analysis_result


gradio_interface_result = gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(type="filepath", label="Audio File"), gr.Dropdown(
        choices=["es", "en"],  label="output Language",  value="en", )],
    outputs="text",
    title="Whisper ",
    description="Real-time demo for speech recognition using a fine-tuned Whisper small model.",
    examples=[
            # Replace with actual file path
            ["/Users/mac/Downloads/source/audio.wav"],
    ],
    # live=True,  # Automatically trigger the interface
)

# Create Summarization Interface
Summarization_interface = gr.Interface(
    fn=summarization_function,
    inputs=None,
    outputs="text",
    live=True,
    title="Summarization Interface",
    description="Summarize text.",
)


# Create Analysis Interface
analysis_interface = gr.Interface(
    fn=analysis_function,
    inputs=None,
    outputs="text",
    live=True,
    title="Analysis Interface",
    description="Perform analysis on text.",
)

# Create Tabbed Interface
app = gr.TabbedInterface(
    interface_list=[gradio_interface_result,
                    Summarization_interface, analysis_interface],
    tab_names=["Transcribe", "Summarization", "Analysis"]
)

if __name__ == "__main__":
    app.launch(share=True,
               server_name="127.0.0.1", server_port=7860)
