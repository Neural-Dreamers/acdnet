import os
import random
import sys
import copy
import math

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
        # return len(self.samples);

    def __getitem__(self, batchIndex):
        # Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex)
        batchX = np.expand_dims(batchX, axis=1)
        batchX = np.expand_dims(batchX, axis=3)
        return batchX, batchY

    def getItems(self):
        n_batches = math.ceil(len(self.data) / self.opt.batchSize)
        random.shuffle(self.data)
        return [self.__getitem__(i) for i in range(n_batches)]

    def generate_batch(self, batchIndex):
        n_batches = math.ceil(len(self.data) / self.opt.batchSize)
        sounds, labels = zip(*(self.data[batchIndex * self.opt.batchSize:]
                               if (n_batches - 1) == batchIndex
                               else self.data[
                                    batchIndex * self.opt.batchSize:(batchIndex + 1) * self.opt.batchSize]))

        sounds = np.asarray(sounds)
        labels = np.asarray(labels)
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
    train_sounds, train_labels = pitch_shift_and_time_stretch(train_sounds, train_labels, options)

    sounds = []
    labels = []

    selected = []

    for i in range(len(train_sounds) * options.mixup_factor):
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

        # Save mixed and preprocessed audio file
        # lab = list(zip(*sorted([(i, v) for i, v in enumerate(label, 1) if v>0], key=lambda x:x[1], reverse=True)))[0]
        # filename = '{}_{}_{}.wav'.format(lab[0], lab[1],i)
        # sf.write(os.path.join("D:\\ACADEMIC\\FYP\\acdnet\\datasets\\fsc22\\generated_sounds", filename), sound,
        #          options.sr)

    return sounds, labels

def pitch_shift_and_time_stretch(train_sounds, train_labels, options):
    sounds = copy.deepcopy(train_sounds)
    labels = copy.deepcopy(train_labels)

    norm = u.normalize(32768.0)
    norm2 = u.normalize(1/32768.0)
    for i in range(0, len(train_sounds)):
        audio_data = train_sounds[i]
        audio = norm(audio_data)

        for k, v in options.augmentation_data.items():
            if k == "time_stretch":
                stretched_audio_data = librosa.effects.time_stretch(audio, rate=v)
                stretched_sound = np.array(stretched_audio_data)
                sounds.append(norm2(stretched_sound))
                labels.append(train_labels[i])
            elif k == "pitch_shift":
                shifted_audio_data = librosa.effects.pitch_shift(audio, sr=options.sr, n_steps=v)
                shifted_sound = np.array(shifted_audio_data)
                sounds.append(norm2(shifted_sound))
                labels.append(train_labels[i])
            else:
                print("Invalid augmentation function")
        sounds[i] = norm2(audio)

    sounds = list(map(lambda x: x.astype(np.float64), sounds))
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

    training_data_path = os.path.join(opt.data, opt.dataset, 'training_data.npz')
    mixed_sounds, mixed_labels = [], []
    if not os.path.exists(training_data_path):
        mixed_sounds, mixed_labels = preprocess_dataset(train_sounds, train_labels, opt)
        training_data = {"sounds":mixed_sounds, "labels":mixed_labels}
        np.savez(training_data_path, **training_data)
    else:
        prepared_data = np.load(training_data_path, allow_pickle=True)
        mixed_sounds, mixed_labels = prepared_data["sounds"], prepared_data["labels"]
    trainGen = Generator(mixed_sounds, mixed_labels, opt)
    print("* {} data ready to train the model".format(len(mixed_sounds)))
    return trainGen
