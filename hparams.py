import argparse

class HParams(object):
    def __init__(self):
        # Feature Parameters
        self.n_channels=128
        self.sample_rate=16000
        self.n_fft=512
        self.f_min=0.0
        self.f_max=8000.0
        self.n_mels=128
        self.n_class=50
        self.n_harmonic=6
        self.semitone_scale=2
        self.learn_bw='only_Q'

        # Training Parameters
        self.device = 1  # 0: CPU, 1: GPU0, 2: GPU1, ...
        self.batch_size = 16
        self.num_epochs = 5
        self.learning_rate = 1e-2
        self.stopping_rate = 1e-5
        self.weight_decay = 1e-6
        self.momentum = 0.9
        self.factor = 0.2
        self.patience = 5
        self.num_workers = 8

    # Function for pasing argument and set hParams
    def parse_argument(self, print_argument=True):
        parser = argparse.ArgumentParser()
        for var in vars(self):
            value = getattr(hparams, var)
            argument = '--' + var
            parser.add_argument(argument, type=type(value), default=value)

        args = parser.parse_args()
        for var in vars(self):
            setattr(hparams, var, getattr(args,var))

        if print_argument:
            print('----------------------')
            print('Hyper Paarameter Settings')
            print('----------------------')
            for var in vars(self):
                value = getattr(hparams, var)
                print(var + ":" + str(value))
            print('----------------------')

hparams = HParams()
hparams.parse_argument()
