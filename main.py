import torch
import numpy as np
import GNet
import MyAudio
import MyText
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import shutil
import os
import uvicorn



#------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def select_device():
    print(f'Selected device for calculations is {torch.cuda.get_device_name()}') if torch.cuda.is_available() else print(f'Selected device for calculations is cpu')
    return 'cuda' if torch.cuda.is_available() else 'cpu'
#------------------------------------------------------------------------------------------------------------------------------------------------------------------#


# default_device = select_device()

# # Load Models
# Gender_Classification_MLP = GNet.ModelLoader(GNet.ModelType.best_epoch, device=default_device)
# ASR_Whisper = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path="openai/whisper-large-v3", torch_dtype=torch.float16).to(default_device)

# print('Models Loaded')

# # Load Processors
# GNet_Processor = GNet.WaveFormProcess()
# Whisper_Processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# print('Processors Loaded')


#--------------------------------------------------------------------------------------------------------------#
# def Load_n_process_audio(file_path, separator_output_directory):
#     """
#     Load and process audio from the given file path.

#     Args:
#         file_path (str): The path to the audio file to be processed.
#         separator_output_directory (str): Directory to save separated audio chunks.

#     Returns:
#         list: A list of processed audio waveforms.
#     """
#     Audio_chunks = MyAudio.pipe(file_path, separator_output_directory)
#     Waveforms = [np.array(chunk.normalize().set_channels(1).get_array_of_samples()) for chunk in Audio_chunks]
#     return Waveforms
#--------------------------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------------------------#
def load_n_process_audio(file_path):
    """
    Load and process audio from the given file path without the background separation.

    Args:
        file_path (str): The path to the input audio file that needs to be processed.

    Returns:
        list: A list of NumPy arrays, where each array represents the waveform data for a chunk of audio.
    """
    Audio_chunks = MyAudio.load_n_split(file_path)
    Waveforms = [np.array(chunk.normalize().set_channels(1).get_array_of_samples()) for chunk in Audio_chunks]
    return Waveforms
#--------------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------------------------------#
def predict(waveforms, language='persian', task='transcribe', sr=16000, float_type=torch.float32):
    """
    Predict transcriptions and gender classification for the given waveforms.

    Args:
        waveforms (list): List of audio waveforms.
        language (str): The language of the audio (default: 'persian').
        task (str): The task for Whisper model (default: 'transcribe').
        sr (int): Sampling rate for audio (default: 16000).

    Returns:
        list: A list of tuples containing transcriptions and gender classifications.
    """

    forced_decoder_ids = Whisper_Processor.get_decoder_prompt_ids(language=language, task=task)
    
    results = []
    for wave in waveforms:

        GNet_Input = GNet_Processor.pipe(wave)
        Whisper_Input = wave/max(wave) # for normalization
        Whisper_Input_Features = Whisper_Processor(Whisper_Input, sampling_rate=sr, return_tensors="pt").input_features
        Whisper_Input_Features = Whisper_Input_Features.to(float_type).to(default_device)

        Whisper_Predicted_ids = ASR_Whisper.generate(Whisper_Input_Features, forced_decoder_ids=forced_decoder_ids)
        Transcription = Whisper_Processor.batch_decode(Whisper_Predicted_ids, skip_special_tokens=True)

        GNet_Output = Gender_Classification_MLP.generate(GNet_Input)
        Gender_Probabilities = (sum(GNet_Output.argmax(-1))/len(GNet_Output)).item()
        Gender = 'male speaker' if 1>=Gender_Probabilities>=0.6 else ('female speaker' if 0<=Gender_Probabilities<=0.4 else 'U')

        results.append([Transcription[0], Gender])
    
    return results
#------------------------------------------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------------------------------#
def pipeline(file_path, 
             whisper_language='persian', 
             whisper_task='transcribe', 
             sr=16000, 
             float_type=torch.float32):
    """
    Main processing pipeline to handle audio input and generate transcriptions with gender classification.

    Args:
        file_path (str): The path to the audio file to be processed.
        whisper_language (str): The language for Whisper model (default: 'persian').
        whisper_task (str): The task for Whisper model (default: 'transcribe').
        sr (int): Sampling rate for audio (default: 16000).
        float_type (torch.dtype): The data type for Torch tensors (default: torch.float32).

    Returns:
        list: A list of tuples containing processed transcriptions and gender classifications.
    """

    # get waveforms from audio files
    Waveforms = load_n_process_audio(file_path)


    # generate transcriptions with gender of the speaker
    Results = predict(Waveforms, language=whisper_language, task=whisper_task, sr=sr, float_type=float_type)

    # correct spell errors and normalize text
    for i, result in enumerate(Results):
        Results[i][0] = MyText.text_pipe(result[0])
    
    return Results
#--------------------------------------------------------------------------------------------------------------------#


if __name__ == "__main__":

    default_device = select_device()
    
    # Load Models
    Gender_Classification_MLP = GNet.ModelLoader(GNet.ModelType.best_epoch, device=default_device)
    ASR_Whisper = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path="openai/whisper-large-v3", torch_dtype=torch.float16).to(default_device)

    print('Models Loaded')

    # Load Processors
    GNet_Processor = GNet.WaveFormProcess()
    Whisper_Processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

    print('Processors Loaded')

    
    # Create API instance
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def root():
        html_content = """
        <html>
            <head>
                <title>API Home</title>
            </head>
            <body>
                <h1>Welcome to the API</h1>
                <p>Access the API documentation <a href="/docs">here</a>.</p>
            </body>
        </html>
        """
        return html_content

    # Define the response model
    class TranscriptionResponse(BaseModel):
        transcription: str
        gender: str

    # The directory where uploaded audio files will be saved
    AUDIO_UPLOAD_DIRECTORY = "./uploaded_audio"

    # Ensure the directories exist
    os.makedirs(AUDIO_UPLOAD_DIRECTORY, exist_ok=True)

    @app.post("/upload_and_process_audio/")
    async def upload_and_process_audio(file: UploadFile = File(...)):
        # Save the uploaded file with its original filename and format
        saved_file_path = os.path.join(AUDIO_UPLOAD_DIRECTORY, file.filename)

        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use the pipeline to process the saved audio file
        results = pipeline(saved_file_path, float_type=torch.float16)

        # Format the response (transcriptions and gender classifications)
        response = [{"transcription": item[0], "gender": item[1]} for item in results]

        return {"results": response}
    

    uvicorn.run(app, host="127.0.0.1", port=8000, timeout_keep_alive=1200)