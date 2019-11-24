import argparse
import os
import glob
import random
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import cv2
import multiprocessing
import torch.nn as nn
import torchvision.models as models


def numeric_sort(files):
    return sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))


class EpicVideoDataset(Dataset):
    """
    This is a "random dataset" which samples videos from kitchen. A bit of a lie in that dataset[idx] does not always return the same item!
    """
    def __init__(self, kitchen_dirs, T=200, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], target_height=224, gpu=True):
        assert len(kitchen_dirs) >= 2, "requires at least 2 kitchens to load from!"
        self._dirs = []
        for kitchen in kitchen_dirs:
            video_list = glob.glob(kitchen + '/*')
            video_list = [glob.glob(v + '/*.jpg') for v in video_list]
            video_list = [numeric_sort(v) for v in video_list if len(v) >= T]            
            self._dirs.append(video_list)
        self._T, self._im_size = T, target_height
        self._mean = np.array(mean).reshape((1, 1, 3)).astype(np.float32)
        self._std = np.array(std).reshape((1, 1, 3)).astype(np.float32)
        self._gpu = gpu
        # self._pool = multiprocessing.Pool(16)

    def __len__(self):
        return len(self._dirs)
    
    def _proc_im(self, im_name, start_w):
        im = cv2.imread(im_name)[:,start_w:start_w+600,::-1]
        im = cv2.resize(im, (self._im_size, self._im_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        im = (im - self._mean) / self._std
        return np.transpose(im, (2, 0, 1))[None].astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chosen_video = random.choice(self._dirs[idx])
        indices = np.sort(np.random.choice(len(chosen_video), size=self._T, replace=False))
        start_w = np.random.randint(0, cv2.imread(chosen_video[0]).shape[1] - 600)
        imgs = [self._proc_im(chosen_video[i], start_w,) for i in indices]
        return np.concatenate(imgs, 0)
        

def train_embedding(epic_dir, save_dir='.'):
    if os.path.exists('{}/data_splits.pkl'.format(save_dir)):
        assignments = pkl.load(open('{}/data_splits.pkl'.format(save_dir), 'rb'))
        train_kitchens = assignments['train']
        val_kitchens = assignments['val']
    else:
        kitchen_folders = glob.glob(epic_dir + 'MiniImages/*')
        random.shuffle(kitchen_folders)
        train_kitchens, val_kitchens = kitchen_folders[4:], kitchen_folders[:4]
        pkl.dump({'train': train_kitchens, 'val': val_kitchens}, open('{}/data_splits.pkl'.format(save_dir), 'wb'))
    
    train_dataset = EpicVideoDataset(train_kitchens, T=200)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.cuda()
    print(model)

    for _ in range(2):
        for imgs in train_loader:
            import pdb; pdb.set_trace()
            imgs = imgs.cuda()
            feat1 = model(imgs[0])[:, :, 0, 0]
            print(feat1.shape)   
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="trains a cyclic embedding")
    parser.add_argument('data_path', help="path to epic base folder")
    args = parser.parse_args()

    train_embedding(args.data_path)
