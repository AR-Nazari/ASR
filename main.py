import numpy as np
import GNet
import MyAudio
import MyText
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


# Load Models
Gender_Classification_MLP = GNet.ModelLoader(GNet.ModelType.best_epoch)
ASR_Whisper = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path="openai/whisper-large-v3")


# Load Procesoors
GNet_Processor = GNet.WaveFormProcess()
Whisper_Processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")


def Load_n_process_audio(file_path, separator_output_directory):
    """
    Load and process audio from the given file path.

    Args:
        file_path (str): The path to the audio file to be processed.
        separator_output_directory (str): Directory to save separated audio chunks.

    Returns:
        list: A list of processed audio waveforms.
    """
    Audio_chunks = MyAudio.pipe(file_path, separator_output_directory)
    Waveforms = [np.array(chunk.normalize().set_channels(1).get_array_of_samples()) for chunk in Audio_chunks]
    return Waveforms

def predict(waveforms, language='persian', task='transcribe', sr=16000):
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

        Whisper_Predicted_ids = ASR_Whisper.generate(Whisper_Input_Features, forced_decoder_ids=forced_decoder_ids)
        Transcription = Whisper_Processor.batch_decode(Whisper_Predicted_ids, skip_special_tokens=True)

        GNet_Output = Gender_Classification_MLP.generate(GNet_Input)
        Gender_Probabilities = (sum(GNet_Output.argmax(-1))/len(GNet_Output)).item()
        Gender = 'male speaker' if 1>=Gender_Probabilities>=0.6 else ('female female' if 0<=Gender_Probabilities<=0.4 else 'U')

        results.append((Transcription[0], Gender))
    
    return results

def pipeline(file_path, separator_output_directory, whisper_language='persian', whisper_task='transcribe', sr=16000):
    """
    Main processing pipeline to handle audio input and generate transcriptions with gender classification.

    Args:
        file_path (str): The path to the audio file to be processed.
        separator_output_directory (str): Directory to save separated audio chunks.
        whisper_language (str): The language for Whisper model (default: 'persian').
        whisper_task (str): The task for Whisper model (default: 'transcribe').
        sr (int): Sampling rate for audio (default: 16000).

    Returns:
        list: A list of tuples containing processed transcriptions and gender classifications.
    """

    # get waveforms from audio files
    Waveforms = Load_n_process_audio(file_path, separator_output_directory)

    # generate transcriptions with gender of the speaker
    Results = predict(Waveforms, language=whisper_language, task=whisper_task, sr=16000)

    # correct spell errors and normalize text
    for i, result in enumerate(Results):
        Results[i][0] = MyText.text_pipe(result[0])
    
    return Results