import torch 
import torch.nn as nn
from enum import Enum
import opensmile
import joblib
import numpy as np

#---------------------------------------------------------------------#
class GNet_MLP(nn.Module):

    def __init__(self):
        super(GNet_MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=988, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=32)
        self.fc6 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(dim=-1) # For binary classification
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.softmax(x)
        return x
#---------------------------------------------------------------------#


#---------------------------#
class ModelType(Enum):
    last_epoch = 'last net'
    best_epoch = 'best net'
    base = 'base'
    user = 'user'
#---------------------------#


#-------------------------------------------------------------------------------------------------------------#
class ModelLoader():
    """
    A utility class for loading and managing a neural network model. This class allows the user to 
    select between a base model, a pretrained model (from the last or best epoch), or a custom 
    user-provided model.

    Args:
        model_type (ModelType, optional): Specifies the type of model to load. 
            Default is ModelType.base. Available options are:
                - ModelType.base: Load the base untrained model.
                - ModelType.last_epoch: Load the model from the last training epoch.
                - ModelType.best_epoch: Load the model with the best validation loss during training.
                - ModelType.user: Load a model from a user-specified path.
        model_path (str, optional): The file path to the model file. Required if model_type is ModelType.user.
        device (str, optional): Specifies the device for model computation (e.g., 'cuda' or 'cpu'). Default is 'cpu'.

    Attributes:
        model (GNet_MLP): The neural network model instance.
        device (str): The device on which to run the model (e.g., 'cuda' or 'cpu').

    Methods:
        load_model(path):
            Loads model weights from a specified file path. If the file is not found, 
            the base model will be used with no pretrained weights.
        
        generate(x):
            Generates predictions using the loaded model. This method sets the model 
            to evaluation mode and disables gradient computation for inference.
    """

    def __init__(self, model_type: ModelType = ModelType.base, model_path: str = None, device='cpu'):
        self.device = device
        self.model = GNet_MLP().to(self.device)
        if model_type == ModelType.last_epoch: self.load_model('last_epoch_net.pth')
        elif model_type == ModelType.best_epoch : self.load_model('best_net.pth')
        elif model_type == ModelType.user : self.load_model(model_path)
        else: print("Using the base network with no pretrained weights")

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device), weights_only=True))
            print(f"Loaded model weights from {path}")
            print(f'Model loaded on {self.device}')
        except FileNotFoundError:
            print(f"Model file {path} not found \nUsing base network")

    def generate(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad(): return self.model(x)
#-------------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------------#
class WaveFormProcess():
    """
    A class for processing audio waveforms, extracting features, and handling audio segmentation.

    Args:
        sr (int, optional): The sampling rate of the audio data. Default is 16000.

    Attributes:
        sr (int): The sampling rate of the audio data.
        smile (opensmile.Smile): An instance of the opensmile library configured for feature extraction.
        scaler (sklearn.preprocessing.MinMaxScaler): A preloaded scaler for normalizing the extracted features.

    Methods:
        feature(waveforms):
            Extracts features from a list of audio waveforms using the opensmile library 
            and scales them using a preloaded scaler.

        split_audio(long_wave, chunk_len=5):
            Splits a long audio waveform into smaller chunks of equal length, padding the last chunk if necessary.

        pipe(long_wave):
            Combines the splitting and feature extraction process into a single pipeline, returning 
            the features as a torch tensor.
    """

    def __init__(self, sr = 16000):
        self.sr = sr
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.emobase,
                                     feature_level=opensmile.FeatureLevel.Functionals,
                                     sampling_rate=self.sr)
        self.scaler = joblib.load('minmax_scaler.pkl')
        
    def feature(self, waveforms):
        if len(waveforms) == 1: 
            feature = self.smile(waveforms[0], self.sr).reshape([len(self.smile.feature_names),])
            return self.scaler.transform(feature.reshape(1, len(feature)))
        else:
            features = [None] * len(waveforms)
            for i, wave in enumerate(waveforms):
                features[i] = self.smile(wave, self.sr).reshape([len(self.smile.feature_names),])            
            return self.scaler.transform(np.array(features))
        
    def split_audio(self, long_wave, chunk_len = 5):
        chunk_len_s = chunk_len * self.sr
        if len(long_wave)<=chunk_len_s: 
            return long_wave.reshape((1,len(long_wave)))
        else:
            chunks = []
            for i in range(0, len(long_wave), chunk_len_s):
                chunk = long_wave[i:i + chunk_len_s]
                chunks.append(chunk)
            if len(chunks[-1]<chunk_len_s):
                padding = np.zeros(chunk_len_s - len(chunks[-1]))
                chunks[-1] = np.concatenate((chunks[-1], padding))
            return np.array(chunks)
        
    def pipe(self, long_wave):
        chunks = self.split_audio(long_wave)
        features = self.feature(chunks)
        return torch.tensor(features)
#-------------------------------------------------------------------------------------------------------------#