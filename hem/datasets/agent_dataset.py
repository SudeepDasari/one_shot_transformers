from torch.utils.data import Dataset
import pickle as pkl
import cv2
import glob
import random
import os
import torch
from hem.datasets.util import resize
from hem.datasets.savers.render_loader import ImageRenderWrapper
import random
import numpy as np


SHUFFLE_RNG = 2843014334
class AgentDemonstrations(Dataset):
    def __init__(self, root_dir, height=224, width=224, normalize=True, T_context=15, T_pair=1, N_pair=1, mode='train', split=[0.9, 0.1]):
        assert all([0 <= s <=1 for s in split]) and sum(split)  == 1, "split not valid!"
        assert mode in ['train', 'val'], "mode should be train or val!"
        assert T_context >= 2 or N_pair > 0, "Must return (s,a) pairs or context!"
        assert T_pair >= 1, "Each state/action pair must have at least 1 time-step"

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
        self._im_dims = (width, height)
        self._normalize = normalize
        self._T_context = T_context
        self._T_pair = T_pair
        self._N_pair = N_pair

    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self._files), "invalid index!"

        traj = ImageRenderWrapper(pkl.load(open(self._files[index], 'rb'))['traj'], self._im_dims[1], self._im_dims[0])
        return self._proc_traj(traj)

    def _proc_traj(self, traj):
        context_frames = []
        if self._T_context:
            context_frames = self._make_context(traj)

        if self._N_pair == 0:
            return {}, context_frames
        elif self._N_pair == 1:
            return self._get_pair(traj), context_frames
        
        base_pair = {}
        for _ in range(self._N_pair):
            for k, v in self._get_pair(traj).items():
                if isinstance(v, dict):
                    value_dict = base_pair.get(k, {})
                    for k1, v1 in v.items():
                        value_dict[k1] = value_dict.get(k1, []) + [v1[None]]
                    base_pair[k] = value_dict
                else:
                    base_pair[k] = base_pair.get(k, []) + [v[None]]
        
        for k in base_pair:
            if isinstance(base_pair[k], dict):
                for k1 in base_pair[k]:
                    base_pair[k][k1] = np.concatenate(base_pair[k][k1], 0)
            else:
                base_pair[k] = np.concatenate(base_pair[k], 0)
        
        return base_pair, context_frames

    def _make_context(self, traj):
        clip = lambda x : int(max(0, min(x, len(traj) - 1)))
        per_bracket = len(traj) / self._T_context
        
        frames = [resize(traj.get(0)['obs']['image'], self._im_dims, self._normalize)[None]]
        for i in range(1, self._T_context - 1):
            n = random.randint(clip(i * per_bracket), clip((i + 1) * per_bracket - 1))
            frames.append(resize(traj.get(n)['obs']['image'], self._im_dims, self._normalize)[None])
        frames.append(resize(traj.get(len(traj) - 1)['obs']['image'], self._im_dims, self._normalize)[None])
        return np.transpose(np.concatenate(frames, 0), (0, 3, 1, 2))

    def _get_pair(self, traj):
        pair = {}
        i = random.randint(0, len(traj) - self._T_pair - 1)
        for j in range(self._T_pair + 1):
            t = traj.get(j + i)
            img = resize(t['obs']['image'], self._im_dims, self._normalize)
            pair['s_{}'.format(j)] = dict(image=np.transpose(img, (2, 0, 1)), state=t['obs']['robot-state'])
            if j:
                pair['a_{}'.format(j)] = t['action'].astype(np.float32)
        return pair

if __name__ == '__main__':
    import time
    import imageio
    from torch.utils.data import DataLoader
    batch_size = 10
    ag = AgentDemonstrations('./test_load', normalize=False)
    loader = DataLoader(ag, batch_size = batch_size, num_workers=8)

    start = time.time()
    timings = []
    for pairs, context in loader:
        timings.append(time.time() - start)
        print(context.shape)

        if len(timings) > 1:
            break
        start = time.time()
    print('avg ex time', sum(timings) / len(timings) / batch_size)

    out = imageio.get_writer('out1.gif')
    for t in range(context.shape[1]):
        frame = [np.transpose(fr, (1, 2, 0)) for fr in context[:, t]]
        frame = np.concatenate(frame, 1)
        out.append_data(frame.astype(np.uint8))
    out.close()
