from torch.utils.data import Dataset
import json
from collections import OrderedDict
import torch
import moviepy.editor as mpy
import os
import numpy as np
import cv2


class SomethingSomething(Dataset):
    def __init__(self, root_dir, height=224,width=375,T=30, pair_by_task=True, mode='train'):
        super(Dataset, self).__init__()
        assert pair_by_task, "only paired loading implemented"
        assert T > 2, "must have at least 2 frames"
        
        self._root_dir = os.path.realpath(os.path.expanduser(root_dir))
        labels = json.load(open(root_dir + '/something-something-v2-{}.json'.format(mode), 'r'))
        self._label_map = OrderedDict()

        for t in labels:
            key = t['template'].lower()
            arr = self._label_map.get(key, [])
            arr.append(int(t['id']))
            self._label_map[key] = arr
        
        keys = list(self._label_map.keys())
        assert all([len(self._label_map[k]) > 1 for k in keys])

        self._start_indexes = [(0, keys[0])]
        for  k in keys[1:]:
            last_index, last_k = self._start_indexes[-1]
            self._start_indexes.append((last_index + len(self._label_map[last_k]) * (len(self._label_map[last_k]) - 1), k))
        self._len = sum([len(self._label_map[k]) * (len(self._label_map[k]) - 1) for k in keys])
        self._T = T
        self._height, self._width = height, width

    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        key_index, index_key = -1,  None
        for start, k in self._start_indexes:
            if index >= start:
                key_index, index_key = index - start, k
            else:
                break
        
        print(index_key)
        first_index = key_index // (len(self._label_map[index_key]) - 1)
        second_index = key_index % (len(self._label_map[index_key]) - 1)
        if second_index >= first_index:
            second_index += 1
        
        first_video = self._load_video(self._label_map[index_key][first_index])
        second_video = self._load_video(self._label_map[index_key][second_index])
        return np.transpose(first_video, (0, 3, 1, 2)), np.transpose(second_video, (0, 3, 1, 2))
    
        
    def _load_video(self, index):
        load_dir = os.path.join(self._root_dir, '{}.webm'.format(index))
        frames = [f for f in mpy.VideoFileClip(load_dir).iter_frames()]
        frames = [frames[0]] + [frames[i] for i in np.sort(np.random.choice(len(frames), size=self._T - 2, replace=self._T > len(frames)))] + [frames[-1]]
        
        resize_method = cv2.INTER_CUBIC
        if frames[0].shape[0] * frames[0].shape[1] > self._height * self._width:
            resize_method = cv2.INTER_AREA
        frames = [cv2.resize(f, (self._width, self._height), interpolation=resize_method)[None] for f in frames]
        return np.concatenate(frames, axis=0)
