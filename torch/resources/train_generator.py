import math
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

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, batchIndex):
        # Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex)
        batchX = np.expand_dims(batchX, axis=1)
        batchX = np.expand_dims(batchX, axis=3)
        return batchX, batchY

    def generate_batch(self, batchIndex):
        n_batches = math.ceil(len(self.data) / self.opt.batchSize)
        sounds, labels = zip(*(self.data[batchIndex*self.opt.batchSize:]
                               if (n_batches-1)==batchIndex
                               else self.data[batchIndex * self.opt.batchSize:(batchIndex + 1) * self.opt.batchSize]))
        return sounds, labels

class Preprocessor:
    def __init__(self, options):
        random.seed(42)
        self.opt = options
        self.preprocess_funcs = self.preprocess_setup()

    def preprocess_setup(self):
        funcs = []
        # if self.opt.strongAugment:
            # funcs += [u.random_scale(1.25)]

        funcs += [u.padding(self.opt.inputLength // 2),
                  u.random_crop(self.opt.inputLength),
                  u.normalize(32768.0)]
        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

def preprocess_dataset(train_sounds, train_labels, options):
    preprocessor = Preprocessor(options)

    sounds = []
    labels = []

    selected = []

    for i in range(len(train_sounds)*options.mixup_factor):
        while True:
            ind1 = random.randint(0, len(train_sounds) - 1)
            ind2 = random.randint(0, len(train_sounds) - 1)

            sound1, label1 = train_sounds[ind1], train_labels[ind1]
            sound2, label2 = train_sounds[ind2], train_labels[ind2]
            if label1 != label2 and "{}{}".format(ind1, ind2) not in selected:
                selected.append("{}{}".format(ind1, ind2))
                break
        sound1 = preprocessor.preprocess(sound1)
        sound2 = preprocessor.preprocess(sound2)

        # Mix two examples
        r = np.array(random.random())
        sound = u.mix(sound1, sound2, r, options.sr).astype(np.float32)
        eye = np.eye(options.nClasses)
        label = (eye[label1 - 1] * r + eye[label2 - 1] * (1 - r)).astype(np.float32)

        # For stronger augmentation
        # sound = u.random_gain(6)(sound).astype(np.float32)

        sounds.append(sound)
        labels.append(label)

        for k, v in options.augmentation_data.items():
            if k == "time_stretch":
                audio_data = np.array(sound)
                stretched_audio_data = librosa.effects.time_stretch(audio_data, rate=v)
                stretched_sound = np.array(stretched_audio_data)
                stretched_sound = u.random_crop(options.inputLength)(stretched_sound)
                sounds.append(stretched_sound)
                labels.append(label)
            elif k == "pitch_shift":
                audio_data = np.array(sound)
                shifted_audio_data = librosa.effects.pitch_shift(audio_data, sr=options.sr, n_steps=v)
                shifted_sound = np.array(shifted_audio_data)
                sounds.append(shifted_sound)
                labels.append(label)
            else:
                print("Invalid augmentation function")

        # Save mixed and preprocessed audio file
        # lab = list(zip(*sorted([(i, v) for i, v in enumerate(label, 1) if v>0], key=lambda x:x[1], reverse=True)))[0]
        # filename = '{}_{}_{}.wav'.format(lab[0], lab[1],i)
        # sf.write(os.path.join("D:\\ACADEMIC\\FYP\\acdnet\\datasets\\fsc22\\generated_sounds", filename), sound,
        #          options.sr)
        # sf.write(os.path.join("D:\\ACADEMIC\\FYP\\acdnet\\datasets\\fsc22\\generated_sounds", filename.replace('.', "st.")), stretched_sound,
        #          options.sr)
        # sf.write(os.path.join("D:\\ACADEMIC\\FYP\\acdnet\\datasets\\fsc22\\generated_sounds", filename.replace('.', "sh.")), shifted_sound,
        #          options.sr)
        # if i == 50:
        #     break
    sounds = np.asarray(sounds)
    labels = np.asarray(labels)
    return sounds, labels

def setup(opt, split):
    print("\n* Setting up the training dataset...")
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.sr // 1000)), allow_pickle=True);
    train_sounds = []
    train_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i != split:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    mixed_sounds, mixed_labels = preprocess_dataset(train_sounds, train_labels, opt)
    trainGen = Generator(mixed_sounds, mixed_labels, opt)

    print("\n* Dataset is ready to train the model!")
    return trainGen
