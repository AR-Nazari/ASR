from pydub import AudioSegment
from pydub.silence import split_on_silence
from spleeter.separator import Separator


def separate_audio(input_file, output_directory, model_id='spleeter:2stems'):
    """
    Separates audio sources from a given input file using Spleeter.

    Args:
        input_file (str): The path to the input audio file.
        output_directory (str): The directory where the separated audio files will be saved.
        model_id (str, optional): The identifier for the Spleeter model. Default is 'spleeter:2stems'.
    """
    separator = Separator(model_id)
    separator.separate_to_file(input_file, output_directory)


def load_audio(file_path, sr=16000):
    """
    Loads an audio file and sets the sampling rate.

    Args:
        file_path (str): The path to the audio file to load.
        sr (int, optional): The sampling rate for the audio. Default is 16000.

    Returns:
        AudioSegment: The loaded and resampled audio segment.
    """
    audio = AudioSegment.from_file(file_path)
    return audio.set_frame_rate(sr)


def pipe(file_path, separator_output_directory):
    """
    Processes an audio file by separating it into different sources, 
    loading the separated vocal track, and splitting it into chunks based on silence.

    Args:
        file_path (str): The path to the input audio file that needs to be processed.
        separator_output_directory (str): The directory where the separated audio files will be saved.

    Returns:
        list: A list of AudioSegment objects, each representing a chunk of audio split by silence.
    """
    separate_audio(file_path, separator_output_directory)
    new_path = './'+''.join(file_path.split('.')[:-1]+['/vocals.wav']) 
    audio = load_audio(new_path)
    chunks = split_on_silence(audio, silence_thresh=audio.dBFS-10, min_silence_len=1000)
    return chunks