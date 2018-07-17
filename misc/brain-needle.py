import numpy as np 
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import lfilter, butter

def get_spec(x, n_fft=1024):
    S = librosa.stft(x, n_fft)
    S = librosa.amplitude_to_db(librosa.magphase(S)[0], ref=np.max)
    # S = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024ft=1024  # log_S = librosa.power_to_db(S, ref=np.max)
    # log_S = librosa.power_to_db(librosa.magphase(S, power=2)[0])
    return S


def plot_spec(S, fs, title='Spectrogram', do_save=0, save_file=None):
    librosa.display.specshow(S, sr=fs, y_axis='mel', x_axis='time')
    plt.title(title)
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()

    if do_save == 1:
        plt.savefig(save_file + '.png')

    plt.show()


# load it
# y, sr = librosa.load("brain-needle-single.wav")
# y, sr = librosa.load("yaurel-single.wav")
y, sr = librosa.load("hr_16k_p225_003.wav")
# x = np.arange(0,len(y))

# plot it
# plt.figure()
# plt.title('Brain Needle waveform')
# plt.plot(x,y)
# plt.show()

# # spectro
s = get_spec(y, n_fft=1024)
plot_spec(s, sr, do_save=1, save_file='special')

# # attempt to denoise
# rec =  librosa.segment.recurrence_matrix(s, mode='affinity', sparse=True)
# new_s = librosa.decompose.nn_filter(s, rec=rec, aggregate=np.average)
# plot_spec(new_s, sr)

# bandpass at 6k
# pinched from here: http://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

print(max(y))
print(min(y))
bottom_end = butter_bandpass_filter(y, 100, 1024, sr)
top_end = butter_bandpass_filter(y, 1024, 8000, sr)

# new specs

top_s = get_spec(top_end, n_fft=1024)
bottom_s = get_spec(bottom_end, n_fft=1024)

plot_spec(top_s, sr)
plot_spec(bottom_s, sr)

librosa.output.write_wav('hr_bott.wav', bottom_end, sr=sr, norm=False)
librosa.output.write_wav('hr_top.wav', top_end, sr=sr, norm=False)