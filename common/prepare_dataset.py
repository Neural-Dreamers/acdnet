"""
 Usages: Dataset preparation code for FSC-22
 Prerequisites: FFmpeg and wget needs to be installed.
"""

import sys
import os
import subprocess

import glob
import zipfile

import numpy as np
import wavio
import wget
import pydub
import shutil


def main():
    mainDir = os.getcwd()
    fsc22_path = os.path.join(mainDir, 'datasets\\fsc22')

    if not os.path.exists(fsc22_path):
        os.mkdir(fsc22_path)

    sr_list = [44100, 20000]
    augmentation_data = {"pitch_shift": -2, "time_stretch": 0.6}

    # Set the URL of the FSC-22 dataset
    url = 'https://storage.googleapis.com/kaggle-data-sets/2483929/4213460/bundle/archive.zip'
    # Set the save location for the dataset
    save_location = fsc22_path

    # Download the dataset
    # wget.download(url, save_location)

    # Unzip the dataset
    zip_file = "archive.zip"
    with zipfile.ZipFile(fsc22_path + "\\" + zip_file, "r") as zip_ref:
        zip_ref.extractall(save_location)

    # Remove the zip file
    # os.remove(zip_file)

    fsc22_master_path = os.path.join(fsc22_path, 'FSC-22-master')

    if not os.path.exists(fsc22_master_path):
        shutil.copytree(os.path.join(fsc22_path, 'Audio Wise V1.0-20220916T202003Z-001'), fsc22_master_path)

    fsc22_master_audio_path = os.path.join(fsc22_master_path, 'audio')

    if not os.path.exists(fsc22_master_audio_path):
        os.rename(os.path.join(fsc22_master_path, 'Audio Wise V1.0'),
                  os.path.join(fsc22_master_path, 'audio'))

    # rename audio files and split into folds
    rename_source_files(fsc22_master_audio_path)

    # Convert sampling rate
    for sr in sr_list:
        convert_sr(os.path.join(fsc22_path, 'FSC-22-master', 'audio'),
                   os.path.join(fsc22_path, 'wav{}'.format(sr // 1000)),
                   sr)

    # Create npz files
    for sr in sr_list:
        src_path = os.path.join(fsc22_path, 'wav{}'.format(sr // 1000))

        create_dataset(src_path, os.path.join(fsc22_path, 'wav{}.npz'.format(sr // 1000)))


def rename_source_files(src_path):
    folds = 5
    audio_file_list = sorted(os.listdir(src_path))

    # set skip to limit the number of samples in the fold to reduce training time
    skip = 0

    for fold in range(1, folds + 1):
        for i in range(fold - 1, len(audio_file_list), folds + skip):
            audio_file = audio_file_list[i]
            label = audio_file.split('_')[0]
            index = audio_file.split('_')[1].split('.')[0]
            new_filename = str(fold) + '-' + index + '-' + label + '.wav'
            os.rename(os.path.join(src_path, audio_file), os.path.join(src_path, new_filename))

    for filename in glob.glob(os.path.join(src_path, "*_*")):
        os.remove(filename)


def convert_sr(src_path, dst_path, sr):
    print('* {} -> {}'.format(src_path, dst_path))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        dst_file = src_file.replace(src_path, dst_path)
        # Create an AudioSegment object
        audio_segment = pydub.AudioSegment.from_file(src_file)

        # Set the audio channels to 1
        audio_segment = audio_segment.set_channels(1)

        # Set the audio sample rate to 16000
        audio_segment = audio_segment.set_frame_rate(sr)

        # Export the file to dst_file
        audio_segment.export(dst_file, format="wav")


def create_dataset(src_path, fsc22_dst_path):
    print('* {} -> {}'.format(src_path, fsc22_dst_path))
    fsc22_dataset = {}

    for fold in range(1, 6):
        fsc22_dataset['fold{}'.format(fold)] = {}
        fsc22_sounds = []
        fsc22_labels = []

        for wav_file in sorted(glob.glob(os.path.join(src_path, '{}-*.wav'.format(fold)))):
            sound = wavio.read(wav_file).data.T[0]
            # start = sound.nonzero()[0].min()
            # end = sound.nonzero()[0].max()
            # sound = sound[start: end + 1] # Remove silent sections
            label = int(os.path.splitext(wav_file)[0].split('-')[-1])
            fsc22_sounds.append(sound)
            fsc22_labels.append(label)

        fsc22_dataset['fold{}'.format(fold)]['sounds'] = fsc22_sounds
        fsc22_dataset['fold{}'.format(fold)]['labels'] = fsc22_labels

    np.savez(fsc22_dst_path, **fsc22_dataset)


if __name__ == '__main__':
    main()
