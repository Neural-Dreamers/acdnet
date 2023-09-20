import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='ACDNet Sound Classification')

    # General settings
    parser.add_argument('--netType', default='ACDNet',  required=False)
    parser.add_argument('--data', default='{}/datasets/'.format(os.getcwd()),  required=False)
    parser.add_argument('--dataset', required=False, default='fsc22', choices=['esc10', 'esc50', 'frog', 'fsc22'])
    parser.add_argument('--BC', default=True, action='store_true', help='BC learning')
    parser.add_argument('--strongAugment', default=True,  action='store_true', help='Add scale and gain augmentation')

    opt = parser.parse_args()

    #Leqarning settings
    opt.batchSize = 256
    opt.weightDecay = 5e-4
    opt.momentum = 0.9
    opt.nEpochs = 2000
    opt.LR = 0.1
    opt.schedule = [0.3, 0.6, 0.9]
    opt.warmup = 10

    #Basic Net Settings
    opt.nClasses = 26
    opt.nFolds = 5
    opt.splits = [i for i in range(1, opt.nFolds + 1)]
    opt.sr = 20000
    opt.inputLength = 30225

    #Test data
    opt.nCrops = 10

    opt.augmentation_data = {"time_stretch": 0.8, "pitch_shift": 1.5}
    opt.mixup_factor = 2

    opt.class_labels = ["Fire", "Rain", "Thunderstorm", "Waterdrops", "Wind", "Silence", "Tree Falling",
                        "Helicopter", "Vehicle Engine", "Axe", "Chainsaw", "Generator", "Handsaw", "Firework",
                        "Gunshot", "Whistling", "Speaking", "Footsteps", "Clapping", "Insect", "Frog",
                        "Bird chirping", "Wing flapping", "Lion", "Wolf howl", "Squirrel"]

    return opt


def display_info(opt):
    print('+------------------------------+')
    print('| {} Sound classification'.format(opt.netType))
    print('+------------------------------+')
    print('| dataset  : {}'.format(opt.dataset))
    print('| nEpochs  : {}'.format(opt.nEpochs))
    print('| LRInit   : {}'.format(opt.LR))
    print('| schedule : {}'.format(opt.schedule))
    print('| warmup   : {}'.format(opt.warmup))
    print('| batchSize: {}'.format(opt.batchSize))
    print('| Splits: {}'.format(opt.splits))
    print('+------------------------------+')