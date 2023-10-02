import glob
import math
import os
import random
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import torch.optim as optim

import torch
import torch.nn as nn

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'common'))
# sys.path.append(os.path.join(os.getcwd(), 'torch/resources'));

import common.opts as opts
import common.utils as u
import resources.calculator as calc
import resources.models as models
import resources.train_generator as train_generator

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


###########################################

class Trainer:
    def __init__(self, options=None):
        self.opt = options
        self.testX = None
        self.testY = None
        self.bestAcc = 0.0
        self.bestAccEpoch = 0
        self.trainGen = train_generator.setup(options, split)

    def Train(self):
        train_start_time = time.time()
        net = None
        if self.opt.model_path != 'ACDNet':
            net_path = self.opt.model_path
            file_paths = glob.glob(net_path)
            if len(file_paths) > 0 and os.path.isfile(file_paths[0]):
                state = torch.load(file_paths[0], map_location=self.opt.device)
                net = models.GetACDNetModel(channel_config=state['config']).to(self.opt.device)
                # net = nn.DataParallel(net)
                if self.opt.retrain:
                    net.load_state_dict(state['weight'])
                print('Model Loaded')
            else:
                print('Model has not been found')
                exit()
        else:
            net = models.GetACDNetModel().to(self.opt.device)
            # net = nn.DataParallel(net)
            print('ACDNet model has been prepared for training')

        calc.summary(net, (1, 1, opt.inputLength))

        training_text = "Re-Training" if self.opt.retrain else "Training from Scratch"
        print("{} has been started. You will see update after finishing every training epoch and validation".format(
            training_text))

        lossFunc = torch.nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.SGD(net.parameters(), lr=self.opt.LR, weight_decay=self.opt.weightDecay,
                              momentum=self.opt.momentum, nesterov=True)

        net = nn.DataParallel(net)

        metrics = {}

        # self.opt.nEpochs = 1957 if self.opt.split == 4 else 2000;
        for epochIdx in range(self.opt.nEpochs):
            metrics['{}'.format(epochIdx)] = {}
            epoch_start_time = time.time()
            optimizer.param_groups[0]['lr'] = self.__get_lr(epochIdx + 1)
            cur_lr = optimizer.param_groups[0]['lr']
            running_loss = 0.0
            running_acc = 0.0
            n_batches = math.ceil(len(self.trainGen.data) / self.opt.batchSize)
            X, Y = self.trainGen.__get_items__(n_batches)
            for batchIdx in range(n_batches):
                # with torch.no_grad():
                # x, y = self.trainGen.__getitem__(batchIdx)
                x = X[batchIdx]
                y = Y[batchIdx]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(x)
                running_acc += (((outputs.data.argmax(dim=1) == y.argmax(dim=1)) * 1).float().mean()).item()
                loss = lossFunc(outputs.log(), y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            tr_acc = (running_acc / n_batches) * 100
            tr_loss = running_loss / n_batches

            # Epoch wise validation
            epoch_train_time = time.time() - epoch_start_time

            net.eval()
            val_acc, val_loss = self.__validate(net, lossFunc)
            # Save best model
            self.__save_model(val_acc, epochIdx, net)
            self.__on_epoch_end(epoch_start_time, epoch_train_time, epochIdx, cur_lr, tr_loss, tr_acc, val_loss,
                                val_acc)

            metrics['{}'.format(epochIdx)]['tr_acc'] = tr_acc
            metrics['{}'.format(epochIdx)]['tr_loss'] = tr_loss
            metrics['{}'.format(epochIdx)]['val_acc'] = val_acc
            metrics['{}'.format(epochIdx)]['val_loss'] = val_loss

            running_loss = 0
            running_acc = 0
            net.train()

        total_time_taken = time.time() - train_start_time
        print("Execution finished in: {}".format(u.to_hms(total_time_taken)))

        self.__save_acc_loss_plot(metrics)

    def load_test_data(self):
        data = np.load(os.path.join(self.opt.data, self.opt.dataset,
                                    'test_data_{}khz/fold{}_test3900.npz'.format(self.opt.sr // 1000, self.opt.split)),
                       allow_pickle=True)
        self.testX = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.opt.device)
        self.testY = torch.tensor(data['y']).to(self.opt.device)

    def __get_lr(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
        decay = sum(epoch > divide_epoch)
        if epoch <= self.opt.warmup:
            decay = 1
        return self.opt.LR * np.power(0.1, decay)

    def __get_batch(self, index):
        x = self.trainX[index * self.opt.batchSize: (index + 1) * self.opt.batchSize]
        y = self.trainY[index * self.opt.batchSize: (index + 1) * self.opt.batchSize]
        return x.to(self.opt.device), y.to(self.opt.device)

    def __validate(self, net, lossFunc):
        if self.testX is None:
            self.load_test_data()

        net.eval()
        with torch.no_grad():
            y_pred = None
            batch_size = (self.opt.batchSize // self.opt.nCrops) * self.opt.nCrops
            for idx in range(math.ceil(len(self.testX) / batch_size)):
                x = self.testX[idx * batch_size: (idx + 1) * batch_size]
                scores = net(x)
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data))

            acc, loss = self.__compute_accuracy(y_pred, self.testY, lossFunc)
        net.train()
        return acc, loss

    # Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        with torch.no_grad():
            # Reshape to shape theme like each sample contains 10 samples, calculate mean and find the indices that
            # has the highest average value for each sample
            if self.opt.nCrops == 1:
                y_pred = y_pred.argmax(dim=1) + 1
                y_target = y_target.argmax(dim=1) + 1
            else:
                y_pred = (y_pred.reshape(y_pred.shape[0] // self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(
                    dim=1).argmax(dim=1) + 1
                y_target = (
                    y_target.reshape(y_target.shape[0] // self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(
                    dim=1).argmax(dim=1) + 1
            acc = (((y_pred == y_target) * 1).float().mean() * 100).item()
            # valLossFunc = torch.nn.KLDivLoss();
            loss = lossFunc(y_pred.float().log(), y_target.float()).item()
            # loss = 0.0;
        return acc, loss

    def __on_epoch_end(self, start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        epoch_time = time.time() - start_time
        val_time = epoch_time - train_time
        line = ('SP-{} Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {}  Loss {:.2f}  Acc {:.2f}% | Val: Loss '
                '{:.2f}  Acc(top1) {:.2f}% | HA {:.2f}@{}\n').format(
            self.opt.split, epochIdx + 1, self.opt.nEpochs, u.to_hms(epoch_time), u.to_hms(train_time),
            u.to_hms(val_time), lr, tr_loss, tr_acc, val_loss, val_acc, self.bestAcc, self.bestAccEpoch)
        # print(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    def __save_acc_loss_plot(self, metrics):
        # Extract time, accuracies, and losses from the metrics dictionary
        epochs = list(metrics.keys())
        tr_acc = [entry['tr_acc'] for entry in metrics.values()]
        tr_loss = [entry['tr_loss'] for entry in metrics.values()]
        val_acc = [entry['val_acc'] for entry in metrics.values()]
        val_loss = [entry['val_loss'] for entry in metrics.values()]

        # Create a figure and axis
        fig, ax1 = plt.subplots()

        # Plot accuracy lines
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color='black')
        ax1.plot(epochs, tr_acc, color='#800000', marker='o', label='Training Accuracy')
        ax1.plot(epochs, val_acc, color='#000075', marker='x', label='Validation Accuracy ')
        ax1.tick_params(axis='y', labelcolor='black')

        # Create a second y-axis for loss lines
        ax2 = ax1.twinx()  # Share the same x-axis
        ax2.set_ylabel('Loss', color='black')
        ax2.plot(epochs, tr_loss, color='#3cb44b', marker='s', label='Training Loss')
        ax2.plot(epochs, val_loss, color='#f58231', marker='^', label='Validation Loss')
        ax2.tick_params(axis='y', labelcolor='black')

        # Add a legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')

        # Set a title
        plt.title('Accuracy and Loss Over Epochs')

        accuracy_matrices_path = os.path.join(os.getcwd(), 'torch\\metrics\\accuracy_matrices')
        curr_datetime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        filename = f'training_metrics-{format(curr_datetime)}.png'

        if not os.path.exists(accuracy_matrices_path):
            os.makedirs(accuracy_matrices_path)

        # Save the plot to the specified folder
        save_path = os.path.join(accuracy_matrices_path, filename)
        plt.savefig(save_path, bbox_inches='tight')

    def __save_model(self, acc, epochIdx, net):
        if acc > self.bestAcc:
            pwd = os.getcwd()
            folder_name = "{}/torch/trained_models/{}_fold{}.pt"
            old_model = folder_name.format(pwd, self.opt.model_name.lower(), self.opt.split)
            if os.path.isfile(old_model):
                os.remove(old_model)
            self.bestAcc = acc
            self.bestAccEpoch = epochIdx + 1
            torch.save({'weight': net.state_dict()},
                       folder_name.format(pwd, self.opt.model_name.lower(), self.opt.split))


def Train(options):
    print('Starting {} model Training for Fold-{}'.format(options.model_name.upper(), options.split))
    opts.display_info(options)
    trainer = Trainer(options)
    trainer.Train()


if __name__ == '__main__':
    opt = opts.parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(opt.device)

    opt.retrain = False
    # valid_training_type = False
    # while not valid_training_type:
    #     train_type = input('Enter an option: \n1. Re-Training\n2. Training from Scratch\n:')
    #     if train_type in ['1', '2']:
    #         opt.retrain = True if train_type == '1' else False
    #         valid_training_type = True

    opt.model_path = "ACDNet"
    print('ACDNet base model will be trained.')
    # valid_path = False
    # while not valid_path:
    #     model_path = input("Enter your pruned model path OR keep it blank to train the base ACDNet model\n:")
    #     opt.model_path = "ACDNet" if model_path == '' else model_path
    #     if model_path == '':
    #         opt.model_path = "ACDNet"
    #         print('ACDNet base model will be trained.')
    #         valid_path = True

        # else:
        #     file_paths = glob.glob(os.path.join(os.getcwd(), model_path))
        #     if len(file_paths) > 0 and os.path.isfile(file_paths[0]):
        #         state = torch.load(file_paths[0], map_location=opt.device)
        #         opt.model_path = file_paths[0]
        #         print('Model has been found at: {}'.format(opt.model_path))
        #         valid_path = True

    opt.model_name = "Augmented_Model_mix2_ps_ts"
    # valid_model_name = False
    # while not valid_model_name:
    #     model_name = input('Enter a name that will be used to save the trained model: ')
    #     if model_name != '':
    #         opt.model_name = model_name
    #         valid_model_name = True

    split = 1
    # valid_fold = False
    # split = None
    # while not valid_fold:
    #     fold = input(
    #         "Which fold do you want your model to be Validated:\n"
    #         " 0. 5-Fold Cross Validation\n"
    #         " 1. Fold-1\n"
    #         " 2. Fold-2\n"
    #         " 3. Fold-3\n"
    #         " 4. Fold-4\n"
    #         " 5. Fold-5\n :")
    #     if fold in ['0', '1', '2', '3', '4', '5']:
    #         split = int(fold)
    #         valid_fold = True

    if split == 0:
        # -- Run for all splits
        for split in opt.splits:
            opt.split = split
            Train(opt)
    else:
        opt.split = split
        Train(opt)
