from torch.utils.data import Dataset
import pickle as pkl
import cv2
import glob
import random
import os
import torch


SHUFFLE_RNG = 2843014334
class AgentDemonstrations(Dataset):
    def __init__(self, root_dir, height=224,width=224,T=30, mode='train', split=[0.9, 0.1]):
        assert all([0 <= s <=1 for s in split]) and sum(split)  == 1, "split not valid!"
        assert mode in ['train', 'val'], "mode should be train or val!"

        shuffle_rng = random.Random(SHUFFLE_RNG)
        all_files = sorted(glob.glob('{}/*.pkl'.format(os.path.expanduser(root_dir))))
        shuffle_rng.shuffle(all_files)
        
        pivot = int(len(all_files) * split[0])
        if mode == 'train':
            files = all_files[:pivot]
        else:
            files = all_files[pivot:]
        assert files

        self._files = files
        self._im_dims = (height, width)
        self._T = T

    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self._files), "invalid index!"

        

if __name__ == '__main__':
    ag = AgentDemonstrations('~/test')
    print(len(ag))
    print('asdf')
