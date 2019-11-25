import torch
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import random
import pickle as pkl
import torch.nn as nn
import cv2
import numpy as np
import tqdm
import multiprocessing
import imageio


def numeric_sort(files):
    return sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))


test_kitchens = pkl.load(open('splits.pkl', 'rb'))['test']
train_kitchens = pkl.load(open('splits.pkl', 'rb'))['train']


resnet152 = models.resnet152(pretrained=True)
resnet152=nn.Sequential(*list(resnet152.children())[:-1])
resnet152.eval()
resnet152.cuda()


frame_names = []
for kitchen in train_kitchens:
    for folder in glob.glob(kitchen + '/*'):
        frame_names.extend(glob.glob(folder + '/*.jpg'))
embeddings = np.zeros((len(frame_names), 2048), dtype=np.float32)


def load_frame(name):
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    f = cv2.imread(name)
    start_w = np.random.randint(f.shape[1] - f.shape[0])
    f = cv2.resize(f[:, start_w:start_w+600, ::-1], (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    f = (f - mean) / std
    return np.transpose(f, (2, 0, 1))[None].astype(np.float32)


pool = multiprocessing.Pool(8)
# batch_size = 40
# for i in tqdm.tqdm(range(int(np.ceil(len(frame_names) / float(batch_size))))):
#     start, end = i * batch_size, min((i + 1) * batch_size, len(frame_names))
#     frames = np.concatenate(pool.map(load_frame, frame_names[start:end]), 0)
#     frames = torch.from_numpy(frames).cuda()
#     embeddings[start:end] = resnet152(frames).cpu().detach().numpy()[:, :, 0, 0]
# pkl.dump({'embeddings':embeddings, 'names':frame_names}, open('train_embeds.pkl', 'wb'))

loaded = pkl.load(open('train_embeds.pkl', 'rb'))
embeddings = loaded['embeddings']
names = loaded['names']

T = 160
lframe = lambda name: cv2.resize(cv2.imread(name)[:, :600, ::-1], (224,224), interpolation=cv2.INTER_AREA)
test_folders = []
for t in test_kitchens:
    test_folders.extend(glob.glob(t + '/*'))
for ctr, t in enumerate(test_folders):
    imgs = numeric_sort(glob.glob(t + '/*.jpg'))
    if len(imgs) < T:
        continue
    imgs = [imgs[i] for i in np.sort(np.random.choice(len(imgs), size=T, replace=False))]
    loaded_imgs = pool.map(load_frame, imgs)
    print('handling', t)
    writer = imageio.get_writer('test{}.gif'.format(ctr), fps=3)
    for name, img in tqdm.tqdm(zip(imgs, loaded_imgs), total=T):
        embed = resnet152(torch.from_numpy(img).cuda()).cpu().detach().numpy()[:, :, 0, 0]
        deltas = np.sum(np.square(embed - embeddings), 1)
        in_order = np.argsort(deltas)

        five_closest = []
        for o in in_order:
            if len(five_closest) >= 5:
                break
            if any([names[o].split('/')[-3] in fv for fv in five_closest]):
                continue
            five_closest.append(names[o])
        writer.append_data(np.concatenate([lframe(name)] + [lframe(fv) for fv in five_closest], 1))
    writer.close()

