import GNet
import MyAudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Load Models
Gender_Classification_MLP = GNet.ModelLoader(GNet.ModelType.best_epoch)
ASR_Whisper = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path="openai/whisper-large-v3")

# Load Procesoors
GNet_Processor = GNet.WaveFormProcess()
Whisper_Processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")


