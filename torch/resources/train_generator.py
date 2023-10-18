import os
import random
import sys

import numpy as np
import torch

from sklearn.preprocessing import normalize

import librosa.display

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'common'))

import common.utils as u


class Generator:
    # Generates data for Keras
    def __init__(self, samples, labels, options):
        random.seed(42)
        # Initialization
        self.data = [(samples[i], labels[i]) for i in range(0, len(samples))]
        self.opt = options
        self.batch_size = options.batchSize
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.data) / self.batch_size))
        # return len(self.samples);

    def __get_items__(self, batches):
        device = torch.device("cuda:0")
        batchesX = []
        batchesY = []

        for i in range(batches):
            batchX, batchY = self.__getitem__(i)
            batchesX.append(batchX)
            batchesY.append(batchY)

        return torch.stack(batchesX).to(device), torch.stack(batchesY).to(device)

    def __getitem__(self, batchIndex):
        # Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex)
        batchX = np.expand_dims(batchX, axis=1)
        batchX = np.expand_dims(batchX, axis=3)

        return torch.tensor(np.moveaxis(batchX, 3, 1)), torch.tensor(batchY)

    def generate_batch(self, batchIndex):
        # Generates data containing batch_size samples
        sounds = []
        labels = []

        selected = []

        if self.opt.mixupFactor == 2:
            for i in range(self.batch_size):
                # Training phase of BC learning
                # Select two training examples
                while True:
                    ind1 = random.randint(0, len(self.data) - 1)
                    ind2 = random.randint(0, len(self.data) - 1)

                    sound1, label1 = self.data[ind1]
                    sound2, label2 = self.data[ind2]

                    if len({label1, label2}) == 2 and "{}-{}".format(ind1, ind2) not in selected:
                        selected.append("{}-{}".format(ind1, ind2))
                        break
                sound1 = self.preprocess(sound1)
                sound2 = self.preprocess(sound2)

                # Mix two examples
                r = np.array(random.random())
                sound = u.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
                eye = np.eye(self.opt.nClasses)
                label = (eye[label1 - 1] * r + eye[label2 - 1] * (1 - r)).astype(np.float32)

                # For stronger augmentation
                sound = u.random_gain(6)(sound).astype(np.float32)

                sound_mel_spect = compute_log_mel_spect(sound, self.opt.sr)
                print(sound_mel_spect.shape)

                sounds.append(sound_mel_spect)
                labels.append(label)
        elif self.opt.mixupFactor == 3:
            for i in range(self.batch_size):
                # Training phase of BC learning
                # Select two training examples
                while True:
                    ind1 = random.randint(0, len(self.data) - 1)
                    ind2 = random.randint(0, len(self.data) - 1)
                    ind3 = random.randint(0, len(self.data) - 1)

                    sound1, label1 = self.data[ind1]
                    sound2, label2 = self.data[ind2]
                    sound3, label3 = self.data[ind3]

                    if len({label1, label2, label3}) == 3 and "{}-{}-{}".format(ind1, ind2, ind3) not in selected:
                        selected.append("{}-{}-{}".format(ind1, ind2, ind3))
                        break
                sound1 = self.preprocess(sound1)
                sound2 = self.preprocess(sound2)
                sound3 = self.preprocess(sound3)

                # Mix three examples
                r = np.array(random.random())
                q = np.array(random.random())
                mix_sound = u.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
                sound = u.mix(mix_sound, sound3, q, self.opt.sr).astype(np.float32)
                eye = np.eye(self.opt.nClasses)
                label = (eye[label1 - 1] * r * q + eye[label2 - 1] * (1 - r) * q + eye[label3 - 1] * (1 - q)).astype(
                    np.float32)

                # For stronger augmentation
                sound = u.random_gain(6)(sound).astype(np.float32)

                sound_mel_spect = compute_log_mel_spect(sound, self.opt.sr)

                sounds.append(sound_mel_spect)
                labels.append(label)
        elif self.opt.mixupFactor == 4:
            for i in range(self.batch_size):
                # Training phase of BC learning
                # Select two training examples
                while True:
                    ind1 = random.randint(0, len(self.data) - 1)
                    ind2 = random.randint(0, len(self.data) - 1)
                    ind3 = random.randint(0, len(self.data) - 1)
                    ind4 = random.randint(0, len(self.data) - 1)

                    sound1, label1 = self.data[ind1]
                    sound2, label2 = self.data[ind2]
                    sound3, label3 = self.data[ind3]
                    sound4, label4 = self.data[ind4]

                    if len({label1, label2, label3, label4}) == 4 and "{}-{}-{}-{}".format(ind1, ind2, ind3, ind4) not in selected:
                        selected.append("{}-{}-{}-{}".format(ind1, ind2, ind3, ind4))
                        break
                sound1 = self.preprocess(sound1)
                sound2 = self.preprocess(sound2)
                sound3 = self.preprocess(sound3)
                sound4 = self.preprocess(sound4)

                # Mix four examples
                r = np.array(random.random())
                q = np.array(random.random())
                p = np.array(random.random())
                mix_sound1 = u.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
                mix_sound2 = u.mix(mix_sound1, sound3, q, self.opt.sr).astype(np.float32)
                sound = u.mix(mix_sound2, sound4, p, self.opt.sr).astype(np.float32)
                eye = np.eye(self.opt.nClasses)
                label = (eye[label1 - 1] * r * q * p + eye[label2 - 1] * (1 - r) * q * p + eye[label3 - 1] * (
                            1 - q) * p + eye[label4 - 1] * (1 - p)).astype(
                    np.float32)

                # For stronger augmentation
                sound = u.random_gain(6)(sound).astype(np.float32)

                sound_mel_spect = compute_log_mel_spect(sound, self.opt.sr)

                sounds.append(sound_mel_spect)
                labels.append(label)

        sounds = np.asarray(sounds)
        labels = np.asarray(labels)

        return sounds, labels

    def preprocess_setup(self):
        funcs = []
        if self.opt.strongAugment:
            funcs += [u.random_scale(1.25)]

        funcs += [u.padding(self.opt.inputLength // 2),
                  u.random_crop(self.opt.inputLength),
                  u.normalize(32768.0)]
        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound


# def preprocess_dataset(train_sounds, train_labels, options):
#     sounds = copy.deepcopy(train_sounds)
#     labels = copy.deepcopy(train_labels)
#
#     norm = u.normalize(32768.0)
#     norm2 = u.normalize(1/32768.0)
#     for i in range(0, len(train_sounds)):
#         audio_data = train_sounds[i]
#         audio = norm(audio_data)
#
#         for k, v in options.augmentation_data.items():
#             if k == "time_stretch":
#                 stretched_audio_data = librosa.effects.time_stretch(audio, rate=v)
#                 stretched_sound = np.array(stretched_audio_data)
#                 sounds.append(norm2(stretched_sound))
#                 labels.append(train_labels[i])
#             elif k == "pitch_shift":
#                 shifted_audio_data = librosa.effects.pitch_shift(audio, sr=options.sr, n_steps=v)
#                 shifted_sound = np.array(shifted_audio_data)
#                 sounds.append(norm2(shifted_sound))
#                 labels.append(train_labels[i])
#             else:
#                 print("Invalid augmentation function")
#         sounds[i] = norm2(audio)
#
#     sounds = list(map(lambda x: x.astype(np.float64), sounds))
#     return sounds, labels

def setup(opt, split):
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.sr // 1000)), allow_pickle=True);
    train_sounds = []
    train_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i != split:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    trainGen = Generator(train_sounds, train_labels, opt)
    print("* {} data ready to train the model".format(len(train_sounds)))
    return trainGen


def compute_log_mel_spect(audio, sample_rate):
    feature_spectrogram = generate_features(audio, sample_rate)
    print(f'feature_spectrogram : {feature_spectrogram.shape}')
    exit(0)
    return feature_spectrogram
    # Mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    # mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram)
    # delta = librosa.feature.delta(mel_spectrogram_db)
    # image = np.dstack((mel_spectrogram_db, delta))
    # print(f'image : {image.shape}')
    # img = Image.fromarray(image, 'RGB')
    # img.save('test_1.png')
    # log_ms = generate_features(audio, sample_rate)
    # print(log_ms.shape)
    # img = Image.fromarray(log_ms, 'RGB')
    # img.save('new_out_2.png')
    # exit(0)
    # fig = plt.figure()
    # canvas = fig.canvas
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax = fig.gca()
    # ax.axis('off')
    #
    # ps = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    # ps_db = librosa.power_to_db(ps, ref=np.max)  # log-mel spectrogram (2d - [n_mels,t])
    # librosa.display.specshow(ps_db, sr=sample_rate)
    # fig.savefig('out1.png')
    # plt.close(fig)
    #
    # print(ps_db.shape)
    # delta = librosa.feature.delta(ps_db)  # delta (2d)
    # delta2 = librosa.feature.delta(ps_db, order=2)  # delta-delta (2d)
    # ps_db = np.expand_dims(ps_db, axis=-1)  # 3d ([n_mels,t,1])
    # delta = np.expand_dims(delta, axis=-1)  # 3d
    # delta2 = np.expand_dims(delta2, axis=-1)  # 3d
    # final_map = np.concatenate([ps_db, delta, delta2], axis=-1)
    # print(final_map.shape)
    # resized_map = np.resize(final_map, [224, 224, 3])
    # img = Image.fromarray(resized_map, 'RGB')
    # img.save('out2.png')
    # print(resized_map.shape)
    # exit(0)
    # librosa.display.specshow(log_ms, sr=sample_rate)
    #
    # canvas.draw()  # Draw the canvas, cache the renderer
    #
    # log_ms_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    # log_ms = log_ms_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    # plt.close(fig)

    # return log_ms


def generate_features(y_cut, sr):
    # my max audio file feature width
    max_size = 1000

    stft = padding(np.abs(librosa.stft(y=y_cut, n_fft=255, hop_length=512)), 128, max_size)
    MFCCs = padding(librosa.feature.mfcc(y=y_cut, n_mfcc=128, sr=sr), 128, max_size)
    spec_centroid = librosa.feature.spectral_centroid(y=y_cut, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y_cut, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y_cut, sr=sr)

    # Now the padding part
    image = np.array([padding(normalize(spec_bw), 1, max_size)]).reshape(1, max_size)
    image = np.append(image, padding(normalize(spec_centroid), 1, max_size), axis=0)

    # repeat the padded spec_bw, spec_centroid and chroma stft until they are stft and MFCC-sized
    for i in range(0, 9):
        image = np.append(image, padding(normalize(spec_bw), 1, max_size), axis=0)
        image = np.append(image, padding(normalize(spec_centroid), 1, max_size), axis=0)
        image = np.append(image, padding(normalize(chroma_stft), 12, max_size), axis=0)

    image = np.dstack((image, np.abs(stft)))
    image = np.dstack((image, MFCCs))

    return image


# def generate_features(y_cut, sr):
    # max_size = 1000  # my max audio file feature width
    # stft = librosa.stft(y_cut, n_fft=255, hop_length=512)
    # print(f'stft : {stft.shape}')
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # librosa.display.specshow(stft, sr=sr)
    # fig.savefig('stft.png')
    # plt.close(fig)

    # mel1 = librosa.feature.melspectrogram(y=y_cut, sr=sr)
    # mel3 = librosa.feature.melspectrogram(y=y_cut, sr=sr)
    # print(f'mel1 : {mel1.shape}')
    # print(mel1)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # librosa.display.specshow(mels, sr=sr)
    # fig.savefig('mels.png')
    # plt.close(fig)

    # spec_centroid = librosa.feature.spectral_centroid(y=y_cut, sr=sr)
    # print(f'spec_centroid : {spec_centroid.shape}')
    #
    # chroma_stft = librosa.feature.chroma_stft(y=y_cut, sr=sr)
    # print(f'chroma_stft : {chroma_stft.shape}')
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # librosa.display.specshow(chroma_stft, sr=sr)
    # fig.savefig('chroma_stft.png')
    # plt.close(fig)
    #
    # spec_bw = librosa.feature.spectral_bandwidth(y=y_cut, sr=sr)
    # print(f'spec_bw : {spec_bw.shape}')
    #
    # # Now the padding part
    # image = np.array([padding(normalize(spec_bw), 1, max_size)]).reshape(1, max_size)
    # print(f'image 1: {image.shape}')
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # librosa.display.specshow(spec_bw, sr=sr)
    # fig.savefig('spec_bw.png')
    # plt.close(fig)
    #
    # image = np.append(image, padding(normalize(spec_centroid), 1, max_size), axis=0)
    # print(f'image 2: {image.shape}')
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # librosa.display.specshow(spec_centroid, sr=sr)
    # fig.savefig('spec_centroid.png')
    # plt.close(fig)

    # exit(0)
    # repeat the padded spec_bw, spec_centroid and chroma stft until they are stft and MFCC-sized
    # for i in range(0, 9):
    #     image = np.append(image, padding(normalize(spec_bw), 1, max_size), axis=0)
    #     image = np.append(image, padding(normalize(spec_centroid), 1, max_size), axis=0)
    #     image = np.append(image, padding(normalize(chroma_stft), 12, max_size), axis=0)
    # image = np.dstack((normalize(mel1), normalize(mel2)))
    # image = np.dstack((image, normalize(mel3)))
    # print(image.shape)
    # print(image)
    # img = Image.fromarray(image, 'RGB')
    # img.save('mel3.png')
    # exit(0)
    # return image


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = max((xx - h) // 2, 0)
    aa = max(0, xx - a - h)
    b = max(0, (yy - w) // 2)
    bb = max(yy - b - w, 0)

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')
