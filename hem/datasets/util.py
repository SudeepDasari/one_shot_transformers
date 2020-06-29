import numpy as np
import cv2
import random
from hem.datasets import Trajectory


SHUFFLE_RNG = 2843014334
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
SAWYER_DEMO_PRIOR = np.array([-1.95254033,  -3.25514605,  1.0,        -2.85691298,  -1.41135844,
                            -1.33966008,  -3.25514405,  1.0,        -2.89548721,  -1.26587143,
                            0.86143858,  -2.36652955,  1.0,        -2.61823206,   0.2176199,
                            -3.54059052,  -4.00911932,  1.0,        -5.07546054,  -5.25952708,
                            -3.7442406,   -5.35087854,  1.0,        -4.2814715,   -3.72755719,
                            -3.85309935,  -5.71775012,  1.0,        -7.02858012,  -3.15234408,
                            -3.67211177, 1.0,         -3.22493535, -12.45458803, -11.76144085, 0, 0])


def resize(image, target_dim, normalize=False, reduce_bits=False):
    if image.shape[:2] != target_dim:
        inter_method = cv2.INTER_AREA
        if np.prod(image.shape[:2]) > np.prod(target_dim):
            inter_method = cv2.INTER_LINEAR
        
        resized = cv2.resize(image, target_dim, interpolation=inter_method)
    else:
        resized = image

    if len(resized.shape) == 2:
        resized = resized[:,:,None]
    
    if reduce_bits:
        assert resized.dtype == np.uint8, "math only works on uint8 data!"
        resized -= resized % 8

    if normalize:
        return (resized.astype(np.float32) / 255 - MEAN) / STD

    return resized.astype(np.float32)


def crop(img, crop):
    if crop[0] > 0:
        img = img[crop[0]:]
    if crop[1] > 0:
        img = img[:-crop[1]]
    if crop[2] > 0:
        img = img[:,crop[2]:]
    if crop[3] > 0:
        img = img[:,:-crop[3]]
    return img


def randomize_video(frames, color_jitter=None, rand_gray=None, rand_crop=None, rand_rot=0, rand_trans=np.array([0,0]), normalize=False):
    frames = [fr for fr in frames]
    
    if color_jitter is not None:
        rand_h, rand_s, rand_v = [np.random.uniform(-h, h) for h in color_jitter]
        delta = np.array([rand_h * 180, rand_s, rand_v]).reshape((1, 1, 3)).astype(np.float32)
        frames = [np.clip(cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_RGB2HSV) + delta, cv2.COLOR_HSV2RGB), 0, 255) for fr in frames]
    if rand_gray and np.random.uniform() < rand_gray:
        frames = [np.tile(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)[:,:,None], (1,1,3)) for fr in frames]
    if rand_crop is not None:
        r1, r2 = [np.random.randint(rand_crop[0]) for _ in range(2)]
        c1, c2 = [np.random.randint(rand_crop[1]) for _ in range(2)]
        
        h, w = frames[0].shape[:2]
        frames = [fr[r1:] for fr in frames]
        frames = [fr[:-r2] for fr in frames] if r2 else frames
        frames = [fr[:,c1:] for fr in frames]
        frames = [fr[:,:-c2] for fr in frames] if c2 else frames
        frames = [resize(fr, (w, h), normalize=False) for fr in frames]
    if rand_rot or any(rand_trans):
        rot = np.random.uniform(-rand_rot, rand_rot)
        trans = np.random.uniform(-rand_trans, rand_trans)
        M = np.array([[np.cos(rot), -np.sin(rot), trans[0]], [np.sin(rot), np.cos(rot), trans[1]]])
        frames = [cv2.warpAffine(fr, M, (fr.shape[1], fr.shape[0])) for fr in frames]
    if normalize:
        frames = [(fr.astype(np.float32) / 255 - MEAN) / STD for fr in frames]
    else:
        for fr in frames:
            fr /= 255
    frames = np.concatenate([fr[None] for fr in frames], 0).astype(np.float32)
    return frames


def split_files(file_len, splits, mode='train'):
    assert sum(splits) == 1 and all([0 <= s for s in splits]), "splits is not valid pdf!"

    order = [i for i in range(file_len)]
    random.Random(SHUFFLE_RNG).shuffle(order)
    pivot = int(len(order) * splits[0])
    if mode == 'train':
        order = order[:pivot]
    else:
        order = order[pivot:]
    return order


def select_random_frames(frames, n_select, sample_sides=False):
    selected_frames = []
    clip = lambda x : int(max(0, min(x, len(frames) - 1)))
    per_bracket = max(len(frames) / n_select, 1)

    for i in range(n_select):
        n = clip(np.random.randint(int(i * per_bracket), int((i + 1) * per_bracket)))
        if sample_sides and i == 0:
            n = 0
        elif sample_sides and i == n_select - 1:
            n = len(frames) - 1
        selected_frames.append(n)

    if isinstance(frames, (list, tuple)):
        return [frames[i] for i in selected_frames]
    elif isinstance(frames, Trajectory):
        return [frames[i]['obs']['image'] for i in selected_frames]
    return frames[selected_frames]
