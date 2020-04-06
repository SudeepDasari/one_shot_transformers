from torch.utils.data import Dataset
import pickle as pkl
import cv2
import glob
import random
import os
import torch
from hem.datasets.util import resize
# from hem.datasets.savers.render_loader import ImageRenderWrapper
import random
import numpy as np
import io
import tqdm


SHUFFLE_RNG = 2843014334
class AgentDemonstrations(Dataset):
    def __init__(self, root_dir, height=224, width=224, depth=False, normalize=True, crop=None, render_dims=None, 
                T_context=15, T_pair=1, N_pair=1, freq=1, mode='train', split=[0.9, 0.1], cache=False):
        assert all([0 <= s <=1 for s in split]) and sum(split)  == 1, "split not valid!"
        assert mode in ['train', 'val'], "mode should be train or val!"
        assert T_context >= 2 or N_pair > 0, "Must return (s,a) pairs or context!"
        assert T_pair >= 1, "Each state/action pair must have at least 1 time-step"

        shuffle_rng = random.Random(SHUFFLE_RNG)
        root_dir = os.path.expanduser(root_dir)
        if 'pkl' not in root_dir:
            root_dir = os.path.join(root_dir, '*.pkl')
        all_files = sorted(glob.glob(root_dir))
        shuffle_rng.shuffle(all_files)
        
        pivot = int(len(all_files) * split[0])
        if mode == 'train':
            files = all_files[:pivot]
        else:
            files = all_files[pivot:]
        assert files

        self._files = files
        self._im_dims = (width, height)
        self._render_dims = tuple(render_dims[::-1]) if render_dims is not None else self._im_dims
        self._crop = tuple(crop) if crop is not None else (0, 0, 0, 0)
        self._depth = depth
        self._normalize = normalize
        self._T_context = T_context
        self._T_pair = T_pair
        self._N_pair = N_pair
        self._freq = freq
        self._cache = {} if cache else None
        for f_name in tqdm.tqdm(self._files):
            with open(f_name, 'rb') as f:
                self._cache[f_name] = f.read()

    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self._files), "invalid index!"

        if self._cache is None:
            traj = pkl.load(open(self._files[index], 'rb'))['traj']
        else:
            f_name = self._files[index]
            if f_name not in self._cache:
                with open(f_name, 'rb') as f:
                    self._cache[f_name] = f.read()
            f = pkl.load(io.BytesIO(self._cache[f_name]))['traj']
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
        
        def _make_frame(n):
            obs = traj.get(n)['obs']
            img = self._crop_and_resize(obs['image'], self._normalize)
            if self._depth:
                img = np.concatenate((img, self._crop_and_resize(obs['depth'][:,:,None], False)), -1)
            return img[None]

        frames = [_make_frame(0)]
        for i in range(1, self._T_context - 1):
            n = random.randint(clip(i * per_bracket), clip((i + 1) * per_bracket - 1))
            frames.append(_make_frame(n))
        frames.append(_make_frame(len(traj) - 1))
        return np.transpose(np.concatenate(frames, 0), (0, 3, 1, 2))

    def _get_pair(self, traj):
        pair = {}
        i = random.randint(0, len(traj) - self._T_pair * self._freq - 1)
        for j in range(self._T_pair + 1):
            t = traj.get(j * self._freq + i)
            img = self._crop_and_resize(t['obs']['image'], self._normalize)
            joint_gripper_state = np.concatenate((t['obs']['joint_pos'], t['obs']['gripper_qpos'])).astype(np.float32)
            pair['s_{}'.format(j)] = dict(image=np.transpose(img, (2, 0, 1)), state=joint_gripper_state)
            if self._depth:
                pair['s_{}'.format(j)]['depth'] = np.transpose(self._crop_and_resize(t['obs']['depth'][:,:,None], False), (2, 0, 1))
            if j:
                pair['a_{}'.format(j)] = t['action'].astype(np.float32)
        return pair
    
    def _crop_and_resize(self, img, normalize):
        if self._crop[0] > 0:
            img = img[self._crop[0]:]
        if self._crop[1] > 0:
            img = img[:-self._crop[1]]
        if self._crop[2] > 0:
            img = img[:,self._crop[2]:]
        if self._crop[3] > 0:
            img = img[:,:-self._crop[3]]
        return resize(img, self._im_dims, normalize)


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
