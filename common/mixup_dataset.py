import gc
import os
import sys
import pickle
from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'common'))

import common.utils as u
import common.opts as opts

opt = opts.parse()
input_length = 100000


def preprocess_setup():
    funcs = []
    if opt.strongAugment:
        funcs += [u.random_scale(1.25)]

    funcs += [u.padding(opt.inputLength // 2),
              u.random_crop(opt.inputLength),
              u.normalize(32768.0)]
    return funcs


preprocess_funcs = preprocess_setup()


def preprocess(sound):
    for f in preprocess_funcs:
        sound = f(sound)

    return sound


def generate_pickle_files(spect_folds):
    pickle_filename = 'mix2_audio_5_20_30225'
    pickle_dir = os.path.join(os.getcwd(), 'datasets/fsc22/Pickle Files')

    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    for fold in range(5):
        filename = pickle_filename

        save_path = os.path.join(pickle_dir, filename)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filename = filename + '_fold' + str(fold + 1)
        save_file = os.path.join(save_path, filename)

        with open(save_file, 'wb') as file:
            pickle.dump(spect_folds[fold], file)


def create_dataset_folds(x, y):
    stratified_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    spect_folds = []

    # Iterate over the folds
    for fold, (train_index, test_index) in enumerate(stratified_5fold.split(x, y)):
        # Split the data into training and testing sets
        x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # Now, X_train, y_train contain the training data for the current fold
        # and X_test, y_test contain the testing data for the current fold

        print(f"Fold {fold + 1}:")
        print(f"Training samples: {len(x_train)}")
        print(f"Testing samples: {len(x_test)}")

        print('Testing bin count -  {}'.format(np.bincount(y_test)))

        test_comp = [list(e) for e in zip(x_test, y_test)]

        spect_folds.append(test_comp)

    print(f'len spect_folds: {len(spect_folds)}')  # num folds
    print(f'len spect_folds[0]: {len(spect_folds[0])}')  # samples in a fold
    print(f'len spect_folds[0][0]: {len(spect_folds[0][0])}')  # elements in a sample - should be 2
    print(f'shape spect_folds[0][0][0]: {np.shape(spect_folds[0][0][0])}')  # spectrogram shape

    generate_pickle_files(spect_folds)


def create_dataset(train_sounds, train_labels):
    train_data = defaultdict(list)

    for key, value in zip(train_labels, train_sounds):
        train_data[key].append(value)

    mixed_sounds = []
    mixed_labels = []

    for label in train_data:
        print(f"Label: {label}")
        for sound1, sound2 in combinations(train_data[label], 2):
            sound1 = preprocess(sound1)
            sound2 = preprocess(sound2)

            sound = u.mix(sound1, sound2, 0.5, opt.sr).astype(np.float32)
            # For stronger augmentation
            sound = u.random_gain(6)(sound).astype(np.float32)

            mixed_sounds.append(sound)
            mixed_labels.append(label)

        print(f'len(mixed_labels): {len(mixed_labels)}')

    # for i in range(len(train_sounds)):
    #     sound1 = train_sounds[i]
    #     label1 = train_labels[i]
    #     sound1 = preprocess(sound1)
    #     for j in range(len(train_sounds)):
    #         if train_labels[j] == label1 and i != j:
    #             sound2 = train_sounds[j]
    #             label2 = train_labels[j]
    #             sound2 = preprocess(sound2)
    #
    #             sound = u.mix(sound1, sound2, 0.5, opt.sr).astype(np.float32)
    #             # For stronger augmentation
    #             sound = u.random_gain(6)(sound).astype(np.float32)
    #
    #             mixed_sounds.append(sound)
    #             mixed_labels.append(label1)
    #
    #     del train_sounds[i]
    #     del train_labels[i]
    #
    #     print(f'len(train_sounds): {len(train_sounds)}')

    mixed_sounds = np.array(mixed_sounds)
    mixed_labels = np.array(mixed_labels)

    print(f'X shape: {np.shape(mixed_sounds)}')
    print(f'y shape: {np.shape(mixed_labels)}')

    create_dataset_folds(mixed_sounds, mixed_labels)


def main():
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.sr // 1000)), allow_pickle=True)
    train_sounds = []
    train_labels = []

    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']

        train_sounds.extend(sounds)
        train_labels.extend(labels)

    del dataset
    gc.collect()

    print(f'len(train_sounds) : {len(train_sounds)}')
    print(f'len(train_sounds[0]) : {len(train_sounds[0])}')

    create_dataset(train_sounds, train_labels)


if __name__ == '__main__':
    mainDir = os.getcwd()

    print(f'input_length: {opt.inputLength}')
    print(f'opt.sr: {opt.sr}')

    main()
