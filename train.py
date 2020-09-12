'''
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
'''
import torch
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)

from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data import get_audio_loader
from src.model.net import HarmonicCNN
from hparams import hparams
# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams):
        self.model =HarmonicCNN()
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams.learning_rate, momentum=hparams.momentum, weight_decay=1e-6, nesterov=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
        self.learning_rate = hparams.learning_rate
        self.stopping_rate = hparams.stopping_rate
        self.device = torch.device("cpu")

        if hparams.device > 0:
            torch.cuda.set_device(hparams.device - 1)
            self.model.cuda(hparams.device - 1)
            self.criterion.cuda(hparams.device - 1)
            self.device = torch.device("cuda:" + str(hparams.device - 1))

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train'):
        self.model.train() if mode is 'train' else self.model.eval()

        epoch_loss = 0
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            prediction = self.model(x)
            loss = self.criterion(prediction, y)
            
            if mode is 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0)*loss.item()
        epoch_loss = epoch_loss/len(dataloader.dataset)

        return epoch_loss

    # Early stopping function for given validation loss
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate

        return stop

def device_name(device):
    if device == 0:
        device_name = 'CPU'
    else:
        device_name = 'GPU:' + str(device - 1)

    return device_name

def main():
    train_loader = get_audio_loader("../dataset/mtat",
                                    batch_size = hparams.batch_size,
                                    split='TRAIN',
                                    input_length=80000,
                                    num_workers = hparams.num_workers)
    valid_loader = get_audio_loader("../dataset/mtat",
                                    batch_size = hparams.batch_size,
                                    split='VALID',
                                    input_length=80000,
                                    num_workers = hparams.num_workers)
    test_loader = get_audio_loader("../dataset/mtat",
                                    batch_size = hparams.batch_size,
                                    split='TEST',
                                    input_length=80000,
                                    num_workers = hparams.num_workers)
    runner = Runner(hparams)

    print('Training on ' + device_name(hparams.device))
    for epoch in range(hparams.num_epochs):
        train_loss = runner.run(train_loader, 'train')
        valid_loss = runner.run(valid_loader, 'eval')
        print(train_loss, valid_loss)
        
        if runner.early_stop(valid_loss, epoch + 1):
            break

    test_loss = runner.run(test_loader, 'eval')
    print("Training Finished")

if __name__ == '__main__':
    main()
