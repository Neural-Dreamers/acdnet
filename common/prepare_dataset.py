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
    url = 'https://storage.googleapis.com/kaggle-data-sets/2483929/4213460/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230902%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230902T114852Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3d8a4c759dec7652e2d2f5a7524dba134df19248a0226fbde4f81fb0a3d2c6bf9f93a4a0c13dbf5622fd42a225c2ac4c10c936cd5fcac0bf01152717ae556b706e7406f21939dea700cf15b40f77e17d6251e38faf6496401687792332e6ad2e4fa63cdbe35d06846019131ce0f0862eeee935c9ac0a6d847a167ab4f518a60a07295573f4bb5fd7589a2d734260a5cb02af85e379c13d7d19228ce764ec91dc3e72df25e96046f16a4962ef755b5a93d6091a4969aed55bad65c82bd8c1ec870aa2d10a4ea043271a18eeb336574b75b7e2fe46f425447a0e901eef68dd654fb09028896999e50987b13b6f1dfb436ea6ad3157b233e3799ded9488153c87a9'

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
            # sound = sound[start: end + 1]  # Remove silent sections
            label = int(os.path.splitext(wav_file)[0].split('-')[-1])
            fsc22_sounds.append(sound)
            fsc22_labels.append(label)

        fsc22_dataset['fold{}'.format(fold)]['sounds'] = fsc22_sounds
        fsc22_dataset['fold{}'.format(fold)]['labels'] = fsc22_labels

    np.savez(fsc22_dst_path, **fsc22_dataset)

if __name__ == '__main__':
    main()
