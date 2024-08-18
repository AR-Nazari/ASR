import numpy as np
import GNet
import MyAudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


# Load Models
Gender_Classification_MLP = GNet.ModelLoader(GNet.ModelType.best_epoch)
ASR_Whisper = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path="openai/whisper-large-v3")


# Load Procesoors
GNet_Processor = GNet.WaveFormProcess()
Whisper_Processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")


def Load_n_process_audio(file_path, separator_output_directory):
    Audio_chunks = MyAudio.pipe(file_path, separator_output_directory)
    Waveforms = [np.array(chunk.normalize().set_channels(1).get_array_of_samples()) for chunk in Audio_chunks]
    return Waveforms