import os
import librosa 
import logging
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--wavin', type=str, help='file path for audio in')
parser.add_argument('--label', type=str, help='label')
parser.add_argument('--type', type=int, default=0, help='do you wanna add noise or convolve impulse? noise=0')
parser.add_argument('--walk', type=int, default=0, help='grab all file names from directory?')
parser.add_argument('--walkdir', type=str, default='sounds', help='which directory to grab from?')
options = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
label =  options.label
what_type= options.type
dirwalk = options.walk
dirdir = options.walkdir

# these two functions were taken from https://github.com/EliosMolina/audio_degrader/ >>>

def mix_with_sound(x, sr, sound_path, snr):
    """ Mix x with sound from sound_path
    Args:
        x (numpy array): Input signal
        sound_path (str): Path of mixing sound. If it does not exist,
                          it checks the path as relative to resources dir
        snr (float): Signal-to-noise ratio
    Returns:
        (numpy array): Output signal
    """
    if not os.path.isfile(sound_path):
        resource_sound_path = os.path.join(os.path.dirname(__file__),
                                           'resources',
                                           sound_path)
        if os.path.isfile(resource_sound_path):
            sound_path = resource_sound_path
    z, sr = librosa.core.load(sound_path, sr=sr, mono=True)
    logging.debug("Mixing with sound {0}".format(sound_path))
    while z.shape[0] < x.shape[0]:  # loop in case noise is shorter than
        z = np.concatenate((z, z), axis=0)
    z = z[0: x.shape[0]]
    rms_z = np.sqrt(np.mean(np.power(z, 2)))
    logging.debug("rms_z: %f" % rms_z)
    rms_x = np.sqrt(np.mean(np.power(x, 2)))
    logging.debug("rms_x: %f" % rms_x)
    snr_linear = 10 ** (snr / 20.0)
    logging.debug("snr , snr_linear: %f, %f" % (snr, snr_linear))
    snr_linear_factor = rms_x / rms_z / snr_linear
    logging.debug("y = x  + z * %f" % snr_linear_factor)
    y = x + z * snr_linear_factor
    rms_y = np.sqrt(np.mean(np.power(y, 2)))
    y = y * rms_x / rms_y
    return y

def convolve(x, sr, ir_path, level=1.0):
    """ Apply convolution to x using given impulse response (as wav file)
    Args:
        x (numpy array): Input signal
        sr (int): Sample rate
        ir_path (string): Path of impulse response file (wav). If it does not
                          exist, it checks the path as relative to resources
                          dir
        level (float): Level of wet/dry signals (1.0=wet)
    Returns:
        (numpy array): Output signal
    """
    if not os.path.isfile(ir_path):
        resource_ir_path = os.path.join(os.path.dirname(__file__),
                                        'resources',
                                        ir_path)
        if os.path.isfile(resource_ir_path):
            ir_path = resource_ir_path
    logging.info('Convolving with %s and level %f' % (ir_path, level))
    x = np.copy(x)
    ir, sr = librosa.core.load(ir_path, sr=sr, mono=True)
    return np.convolve(x, ir, 'full')[0:x.shape[0]] * level + x * (1 - level)

## << end pilfered code

# # # # # # # # #  start me >>>>>>>
"""
example calls: 
python3 audio_deg.py --wavin hr_p225_003.wav --label 'output/testing' --walk 1 --walkdir 'impulse' --type 1
python3 audio_deg.py --wavin hr_p225_003.wav --label 'output/testing' --walk 1 --walkdir 'sounds' --type 0


https://www.ee.columbia.edu/~dpwe/sounds/noise/
http://spib.linse.ufsc.br/noise.html
"""

def add_noise(audio_in, sr, noise_type, snr):
    path_to_noise = 'sounds/' + noise_type + '.wav'        

    noisy = mix_with_sound(audio_in, sr, path_to_noise, snr)
    out = label + '_' + noise_type + '_'+ str(snr) + '.wav'
    librosa.output.write_wav(out, noisy, sr=sr, norm=False)

def conv_impulse(audio_in, sr, impulse, snr):
    path_to_impulse = 'impulse/' + impulse + '.wav'
    conv_out = convolve(audio_in, sr, path_to_impulse, snr)
    out = label + '_' + impulse + '_' + str(snr) + '.wav'

    librosa.output.write_wav(out, conv_out, sr=sr, norm=False)

def pre_emph(audio, c=0.95):
    # pre-emphasis on the audio in, where c is the coefficient between 0 and .9
    # y[n] = x[n] - α*x[n-1]
    # derivative in discrete time domain 
    # high value means rapid change i.e. high freq
    
    emph_out = np.append(audio[0], audio[1:] - (c * audio[:-1]))

    out = label + '_' + 'pre.wav'
    librosa.output.write_wav(out, emph_out, sr=sr, norm=False)


if __name__ == '__main__':
    x, sr = librosa.core.load(options.wavin, sr=None, mono=False)

    # noise = list()
    # if dirwalk==1:
    #     files = os.listdir(dirdir)
    #     print('yew', files)
    #     for entry in files:
    #         print(entry)
    #         noise.append(entry[:-4]) # should give us every filename without .wav 

    # for noiz in noise:
    #     if what_type == 0:
    #         for snrz in range(20, 26, 2):
    #             add_noise(x, sr, str(noiz), snrz)
    #     else:
    #         conv_impulse(x, sr, str(noiz), 0.5)

    pre_emph(x)
    