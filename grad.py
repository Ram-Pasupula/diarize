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

whisper = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH_W,
    device=device,
    torch_dtype=torch_dtype
)

def transcribe(audio_path):
    text = whisper(audio_path)["text"]
    return text

def summarization_function(input_text):
    summarized_text = f"Summarized text: {input_text}"
    return summarized_text

def analysis_function(input_text):
    analysis_result = f"Analysis result for: {input_text}"
    return analysis_result

def launch_summarization_interface(result_from_gradio):
    summarization_iface = gr.Interface(
        fn=summarization_function,
        inputs="text",
        outputs="text",
        live=True,
        title="Summarization Interface",
        description="Summarize text.",
        examples=[
            [result_from_gradio],  # Use the result from the previous interface
        ]
    )

    # Launch summarization interface
    summarization_iface.launch()

def launch_analysis_interface(result_from_gradio):
    analysis_iface = gr.Interface(
        fn=analysis_function,
        inputs="text",
        outputs="text",
        live=True,
        title="Analysis Interface",
        description="Perform analysis on text.",
        examples=[
            [result_from_gradio],  # Use the result from the previous interface
        ]
    )

    # Launch analysis interface
    analysis_iface.launch()

if __name__ == "__main__":
    # Launch speech recognition interface
    gradio_interface_result = gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(type="filepath", label="Speak into the microphone"),
        outputs="text",
        title="Whisper Small",
        description="Real-time demo for speech recognition using a fine-tuned Whisper small model.",
        examples=[
            ["/Users/mac/Downloads/source/audio.wav"],  # Replace with actual file path
        ],
       # live=True,  # Automatically trigger the interface
    ).launch()

    # Obtain the result from the user input (not the result of launching the interface)
    #user_input_result = input("Enter the result from the user input: ")
   
    demo = gr.Interface(fn=summarization_function, input=gradio_interface_result, outputs="textbox")
    demo.launch()  
