import os
import random
import sys

import numpy as np
import soundfile as sf
import librosa

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

    def __getitem__(self, batchIndex):
        # Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex)
        batchX = np.expand_dims(batchX, axis=1)
        batchX = np.expand_dims(batchX, axis=3)
        return batchX, batchY

    def generate_batch(self, batchIndex):
        # Generates data containing batch_size samples
        sounds = []
        labels = []

        selected = []

        for i in range(self.batch_size*2):
            # Training phase of BC learning
            # Select two training examples
            while True:
                ind1 = random.randint(0, len(self.data) - 1)
                ind2 = random.randint(0, len(self.data) - 1)

                sound1, label1 = self.data[ind1]
                sound2, label2 = self.data[ind2]
                if label1 != label2 and "{}{}".format(ind1, ind2) not in selected:
                    selected.append("{}{}".format(ind1, ind2))
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            r = np.array(random.random())
            sound = u.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
            eye = np.eye(self.opt.nClasses)
            label = (eye[label1 - 1] * r + eye[label2-1] * (1 - r)).astype(np.float32)

            # For stronger augmentation
            sound = u.random_gain(6)(sound).astype(np.float32)

            sounds.append(sound)
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

def preprocess_dataset(train_sounds, train_labels, options, temp_path):
    sounds = train_sounds
    labels = train_labels

    for i, audio_data in enumerate(train_sounds):
        sf.write(temp_path, audio_data, options.sr)
        sound_wave, sample_rate = librosa.load(temp_path)
        for k, v in options.augmentation_data.items():
            if k == "time_stretch":
                stretched_audio_data = librosa.effects.time_stretch(sound_wave, rate=v)
                stretched_sound = np.array(stretched_audio_data)
                sounds.append(stretched_sound)
                labels.append(train_labels[i])
            elif k == "pitch_shift":
                shifted_audio_data = librosa.effects.pitch_shift(sound_wave, sr=sample_rate, n_steps=v)
                shifted_sound = np.array(shifted_audio_data)
                sounds.append(shifted_sound)
                labels.append(train_labels[i])
            else:
                print("Invalid augmentation function")
        os.remove(temp_path)

    sounds = np.asarray(sounds)
    labels = np.asarray(labels)
    return sounds, labels

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

    preprocessed_sounds, preprocessed_labels = preprocess_dataset(train_sounds, train_labels, opt,
                                                                  os.path.join(opt.data, opt.dataset, "temp.wav"))
    trainGen = Generator(preprocessed_sounds, preprocessed_labels, opt)

    return trainGen
