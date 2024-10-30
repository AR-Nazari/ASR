import numpy as np
import librosa


def split_audio(long_wave, chunk_len = 5, sr=16000):
    chunk_len_s = chunk_len * sr
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
    
def classify_f0(chuncks, sr=16000):
    mean_f0s = np.zeros(len(chuncks))
    for i, wave in enumerate(chuncks):
        f0, voiced_flag, _ = librosa.pyin(wave, fmin=50, fmax=300, sr=sr)
        mean_f0s[i] = f0[voiced_flag].mean() if f0[voiced_flag].size > 0 else 0
    mean = mean_f0s[mean_f0s!=0].mean()
    if mean >=180: return "female"
    elif mean <=170: return "male"
    else: "uncertain"

