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
from torch.utils.tensorboard import SummaryWriter


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
        

def train_embedding(epic_dir, T=200, save_dir='.', n_gpus=1, lr=1e-4, n_epochs=10000, lambda_var=0.001, ex_per_step=10):
    if os.path.exists('{}/data_splits.pkl'.format(save_dir)):
        assignments = pkl.load(open('{}/data_splits.pkl'.format(save_dir), 'rb'))
        train_kitchens = assignments['train']
        val_kitchens = assignments['val']
    else:
        kitchen_folders = glob.glob(epic_dir + 'MiniImages/*')
        random.shuffle(kitchen_folders)
        train_kitchens, val_kitchens = kitchen_folders[4:], kitchen_folders[:4]
        pkl.dump({'train': train_kitchens, 'val': val_kitchens}, open('{}/data_splits.pkl'.format(save_dir), 'wb'))
    
    train_dataset = EpicVideoDataset(train_kitchens, T=T)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr)
    optimizer.zero_grad()
    cross_entropy_loss = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    ctr = 0
    for e in range(n_epochs * ex_per_step):
        epoch_loss =  0
        for imgs in train_loader:
            imgs = imgs.cuda()
            U = model(imgs[0])[:, :, 0, 0]
            V = model(imgs[1])[:, :, 0, 0]

            chosen_i = np.random.randint(T)
            deltas = torch.sum((U[chosen_i][None] - V) ** 2, dim=1)
            v_hat = torch.sum(torch.nn.functional.softmax(-deltas, dim=0)[:, None] * V, dim=0)[None]
            class_logits = -torch.sum((v_hat - U) ** 2, dim=1)
            l_1 = cross_entropy_loss(class_logits.view((1, -1)), torch.from_numpy(np.array([chosen_i])).cuda())
            
            betas = torch.nn.functional.softmax(class_logits, dim=0)
            mu = torch.sum(betas * torch.range(0, T-1).cuda())
            sigma_square = torch.sum(betas * (torch.range(0, T-1).cuda() - chosen_i) ** 2)
            l_2 = (chosen_i - mu) / sigma_square + lambda_var * torch.log(sigma_square) / 2

            loss = l_1 + l_2
            loss.backward()

            writer.add_scalar('Loss/pred_loss', l_1.item(), ctr)
            writer.add_scalar('Loss/gauss_loss', l_2.item(), ctr)

            ctr += 1
            epoch_loss += loss.item() / len(train_loader)
            if ctr % ex_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()

        writer.add_scalar('Loss/train', epoch_loss, e)
        writer.file_writer.flush()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="trains a cyclic embedding")
    parser.add_argument('data_path', help="path to epic base folder")
    parser.add_argument('--T', help="number of timesteps to sample from each video", default=70, type=int)
    parser.add_argument('--n_gpus', help="number of gpus to train on", default=1, type=int)
    parser.add_argument('--save_dir', help="number of gpus to train on", default='.', type=str)
    args = parser.parse_args()

    train_embedding(args.data_path, args.T, args.save_dir, args.n_gpus)
