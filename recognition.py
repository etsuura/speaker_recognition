import os
import wave
import itertools
import random

from sklearn.preprocessing import MinMaxScaler
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pyworld as pw
import pysptk
from nnmnkwii.preprocessing import trim_zeros_frames

from layers import Region, Layer, createEncoder, createClassifier
from sdr_util import getDenseArray
import viz_util
import param


def get_wavfile_list(path):
    wav_files = []
    for dirpath, subdirs, files in os.walk(path):
        for x in files:
            if x.endswith(".wav"):
                wav_files.append(os.path.join(dirpath, x))
    return wav_files

def get_wave_data(filename, print_info=False):
    with wave.open(filename , "r") as wf:
        fs = wf.getframerate()  # サンプリング周波数
        sig = wf.readframes(wf.getnframes())
        sig = np.frombuffer(sig, dtype= "int16") / 32768.0  # -1 - +1に正規化
        if print_info:
            print_wave_info(wf)
    return sig, fs

def print_wave_info(wf):
    """WAVEファイルの情報を取得"""
    print ("チャンネル数:", wf.getnchannels())
    print ("サンプル幅:", wf.getsampwidth())
    print ("サンプリング周波数:", wf.getframerate())
    print ("フレーム数:", wf.getnframes())
    print ("パラメータ:", wf.getparams())
    print ("長さ（秒）:", float(wf.getnframes()) / wf.getframerate())

# def normalize_sig(sig):
#     scaler = MinMaxScaler()
#     sig = sig.reshape(-1, 1)
#     sig = scaler.fit_transform(sig)
#     return sig.reshape(-1)
#
# def normalize_spec(spec, min_level_db=-100):
#     return np.clip((spec - min_level_db) / -min_level_db, 0, 1)
#
# def amp_to_db(x):
#     return 20 * np.log10(np.maximum(1e-5, x))
#
# def build_mel_basis(fs):
#     args = {
#         'sr': fs,
#         'n_fft': 2048,
#         'n_mels': 20,
#         'fmin': 40
#     }
#     return librosa.filters.mel(**args)
#
# def linear_to_mel(spectrogram, mel_basis=None):
#     if mel_basis is None:
#         mel_basis = build_mel_basis(fs)
#     return np.dot(mel_basis, spectrogram)
#
# def mel_spectrogram(sig, fs):
#     args = {
#         'y': sig,
#         'n_fft': 2048,
#         'hop_length': int(fs * 0.0125), # 12.5ms
#         'win_length': int(fs * 0.05)   # 50ms
#     }
#     spectrum = librosa.stft(**args)
#     db = amp_to_db(linear_to_mel(np.abs(spectrum)))
#     return normalize_spec(db)
#
# def show_spectrogram(mel, save=False):
#     plt.imshow(mel)
#     plt.show()
#     if save:
#         plt.savefig('mel_spectrogram.png')

class prepare_param():
    def __init__(self):
        self.fs = None
        self.fftlen = None
        self.alpha = None
        self.order = None
        self.frame_period = None
        self.hop_length = None
        self.max_files = None
        self.test_size = None
        self.use_delta = None
        self.window = None
        self.fo = None
        self.sp = None
        self.ap = None
        self.mcep = None

    def set_param(self, path):
        self.data, self.fs = get_wave_data(path)
        self.fftlen = pw.get_cheaptrick_fft_size(self.fs)
        self.alpha = pysptk.util.mcepalpha(self.fs)
        self.order = 20
        self.frame_period = 5
        self.hop_length = int(self.fs * (self.frame_period * 0.001))
        self.max_files = 100  # number of utterances to be used.
        self.test_size = 0.03
        self.use_delta = True

        if self.use_delta:
            self.windows = [
                (0, 0, np.array([1.0])),
                (1, 1, np.array([-0.5, 0.0, 0.5])),
                (1, 1, np.array([1.0, -2.0, 1.0])),
            ]
        else:
            self.windows = [
                (0, 0, np.array([1.0])),
            ]

    def collecr_features(self):
        data = self.data.astype(np.float64)
        _fo, _time = pw.dio(self.data, self.fs)
        self.fo = pw.stonemask(self.data, _fo, _time, self.fs)
        sp = pw.cheaptrick(self.data, self.fo, _time, self.fs)
        self.sp = trim_zeros_frames(sp)
        self.ap = pw.d4c(self.data, self.fo, _time, self.fs)
        self.mcep = pysptk.sp2mc(sp, order=self.order, alpha=self.alpha)
        return self.mcep

def train(train_wav_files, clf, model, encoder, width, speaker_dict):
    i = 0
    for wav_file in train_wav_files:
        answer, prediction = [], []

        dataclass = prepare_param()
        dataclass.set_param(wav_file)
        mels = dataclass.collecr_features()

        # sig, fs = get_wave_data(wav_file)
        # sig = normalize_sig(sig)
        # mels = mel_spectrogram(sig, fs)

        for mel in mels:
            encoding = getDenseArray(mel[1:], encoder, width=width)
            outputs = model.foward(encoding)
            output = outputs[-1][0]

            outputs = list(itertools.chain.from_iterable(outputs))
            outputs = [output for output in outputs if output is not None]

            viz_util.visualize(i, wav_file, encoding, outputs)

            ans = 0
            for speaker in speaker_dict.keys():
                if speaker in wav_file:
                    ans = speaker_dict[speaker]

            clf.learn(output, ans)
            answer.append(ans)
            i += 1

        print(wav_file)
        model.reset()
        # print('answer:', answer)

    print("Train Finish!")

def test(test_wav_files, clf, model, encoder, width, speaker_dict):
    i = 0
    for wav_file in test_wav_files:
        answer, prediction = [], []

        # sig, fs = get_wave_data(wav_file)
        # sig = normalize_sig(sig)
        # mels = mel_spectrogram(sig, fs)

        dataclass = prepare_param()
        dataclass.set_param(wav_file)
        mels = dataclass.collecr_features()

        for mel in mels:
            encoding = getDenseArray(mel[1:], encoder, width=width)
            outputs = model.foward(encoding)
            output = outputs[-1][0]

            outputs = list(itertools.chain.from_iterable(outputs))
            outputs = [output for output in outputs if output is not None]

            viz_util.visualize(i, wav_file, encoding, outputs)

            ans = 0
            for speaker in speaker_dict.keys():
                if speaker in wav_file:
                    ans = speaker_dict[speaker]

            pred = np.argmax(clf.infer(output))
            answer.append(ans)
            prediction.append(pred)

            i += 1

        print(wav_file)


        print('answer:', answer)
        print('prediction:', prediction)
        print('accuracy:', np.sum(np.array(answer) == np.array(prediction)))
        print("")

def main():
    Mode = "RedDot"

    TrainDir = "train"
    TestDir = "test"

    data_path = param.input_file
    train_path = os.path.join(data_path, TrainDir)
    test_path = os.path.join(data_path, TestDir)

    train_wav_files = get_wavfile_list(train_path)
    test_wav_files = get_wavfile_list(test_path)

    random.seed(3)
    random.shuffle(train_wav_files)
    random.shuffle(test_wav_files)

    width = 40
    encoder = createEncoder(width=width)

    model = Region(
        Layer(din=(20, width), dout=(20, 20), temporal=True),
        Layer(din=(20, 20), dout=(10, 10), temporal=False)
    )
    model.compile()

    clf = createClassifier()

    #Todo use dir name
    if Mode == "jvs":
        speaker_dict = {
            'jvs001': 1,
            'jvs002': 2
        }
    elif Mode == "RedDot":
        speaker_dict = {
            'm0001' : 1,
            'f0002' : 2,
            'other' : 3
        }


    train(train_wav_files, clf, model, encoder, width, speaker_dict)
    # model.save()
    test(test_wav_files, clf, model, encoder, width, speaker_dict)

    # i = 0
    # for wav_file in wav_files:
    #     answer, prediction = [], []
    #
    #     sig, fs = get_wave_data(wav_file)
    #     sig = normalize_sig(sig)
    #     mels = mel_spectrogram(sig, fs)
    #
    #     for mel in mels.T:
    #         encoding = getDenseArray(mel, encoder, width=width)
    #         outputs = model.foward(encoding)
    #         output = outputs[-1][0]
    #
    #         outputs = list(itertools.chain.from_iterable(outputs))
    #         outputs = [output for output in outputs if output is not None]
    #
    #         viz_util.visualize(i, wav_file, encoding, outputs)
    #
    #         ans = 0
    #         for speaker in speaker_dict.keys():
    #             if speaker in wav_file:
    #                 ans = speaker_dict[speaker]
    #
    #         if 'test' in wav_file:
    #             pred = np.argmax(clf.infer(output))
    #             answer.append(ans)
    #             prediction.append(pred)
    #         else:
    #             clf.learn(output, ans)
    #
    #         i += 1
    #
    #     print(wav_file)
    #
    #     if 'test' in wav_file:
    #         print('answer:', answer)
    #         print('prediction:', prediction)
    #         print('accuracy:', np.sum(np.array(answer) == np.array(prediction)))

if __name__ == '__main__':
    main()