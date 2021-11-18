import argparse
import os
import sys

class Options():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--eeg_dataset', type=str, default='data/eeg_signals_128_sequential_band_all_with_mean_std.pth', help="EEG dataset path")
        self._parser.add_argument('--splits_path', type=str, default="data/splits_by_image.pth", help="splits path")
        self._parser.add_argument('--split_num', type=int, default=0, help="split number")
        self._parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        self._parser.add_argument('--optim', type=str, default='Adam', help='optimizer')

        self._parser.add_argument('--train_mode', type=str, default='full', help='training mode: full/window/channel')
        self._parser.add_argument('--classifier', type=str, required=True, help="LSTM/MLP/CNN")
        
        self._parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
        self._parser.add_argument('--epochs', default=100, type=int, help='training epochs')
        self._parser.add_argument('--GPUindex', default=0, type=int, help='which GPU to use')


        self._parser.add_argument('--no_cuda', default=False, help="disable CUDA", action="store_true")

        self._parser.add_argument('--window_len', default=200, type=int, help='the length of the window')
        self._parser.add_argument('--window_s', default=0, type=int, help='the starting point of the window')
        self._parser.add_argument('--channel_idx', default=0, type=int, help='the idx of the channel') 
        
        
        self._parser.add_argument('--save_path', type=str, default='checkpoints', help='the path to save trained models')
        

        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()
        self.set_train_mode_param()

        if not os.path.exists(self._opt.save_path):
            try:
                os.makedirs(self._opt.save_path)
            except:
                print('Invalid path to save models! Quit!')
                sys.exit()

        # set is train or set
        self._opt.is_train = self.is_train

        # get and set gpus
        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        return self._opt, self.train_mode_param[self._opt.train_mode]

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def set_train_mode_param(self):
    
        self.train_mode_param = {'full':{'channel_idx':None,
                                         'channel_num':128, 
                                         'eeg_length':500},
                                         
                                 'window':{'channel_idx':None,
                                           'channel_num':128, 
                                           'eeg_length':200},

                                 'channel':{'channel_idx':self._opt.channel_idx,
                                           'channel_num':1, 
                                           'eeg_length':500},}
