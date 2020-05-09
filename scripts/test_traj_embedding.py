import pickle as pkl
from hem.datasets.util import resize, MEAN, STD, crop
from hem.datasets import get_dataset
import torch
import os
from hem.util import parse_basic_config
import glob
import numpy as np
from torch.utils.data import DataLoader
import tqdm
import imageio


def _test_files(test_folder, config):
    im_dims = (config.get('width', 224), config.get('height', 224))
    crop_params = config.get('crop', [0,0,0,0])
    normalize = config.get('normalize', True)
    T = config.get('T_context', 15)

    test_trajs = glob.glob(os.path.expanduser(test_folder) +  '*.pkl')
    for traj in test_trajs:
        traj = pkl.load(open(traj, 'rb'))['traj']
        stride = max(int(len(traj) // T), 1)
        times = [0] + [max(min(t * stride + np.random.randint(stride), len(traj) - 1), 0) for t in range(1, T - 1)] + [len(traj) - 1]
        imgs = [resize(crop(traj[t]['obs']['image'], crop_params), im_dims, normalize)[None] for t in times]
        imgs = np.transpose(np.concatenate(imgs, 0), (0, 3, 1, 2))[None]
        yield imgs.astype(np.float32)


def test_traj_embedding(model, device, source_loader, test_data, save_dir):
    embeds, imgs = [], []
    for b in tqdm.tqdm(source_loader):
        if isinstance(b, (tuple, list)):
            b = b[1]
        with torch.no_grad():
            embeds.append(torch.nn.functional.normalize(model(b.to(device)), dim=1).detach().cpu().numpy())
        vid = np.transpose(b.cpu().numpy(), (0, 1, 3, 4, 2)) * STD[None][None] + MEAN[None][None]
        imgs.append((vid * 255).astype(np.uint8))
    embeds, imgs = np.concatenate(embeds, 0), np.concatenate(imgs, 0)

    test_embeds, test_imgs = [], []
    for b in tqdm.tqdm(test_data):
        b = torch.from_numpy(b).to(device)
        with torch.no_grad():
            test_embeds.append(torch.nn.functional.normalize(model(b.to(device)), dim=1).detach().cpu().numpy())
        vid = np.transpose(b.cpu().numpy(), (0, 1, 3, 4, 2)) * STD[None][None] + MEAN[None][None]
        test_imgs.append((vid * 255).astype(np.uint8))
    test_embeds, test_imgs = np.concatenate(test_embeds, 0), np.concatenate(test_imgs, 0)
    scores = test_embeds.dot(embeds.T)
    
    for s in tqdm.tqdm(range(scores.shape[0])):
        top_5 = np.argsort(scores[s])[-5:][::-1]
        writer = imageio.get_writer(os.path.join(save_dir, 'test{}.gif'.format(s)), fps=10)
        pad = np.zeros((imgs.shape[2], 2, 3)).astype(np.uint8)
        for t in range(test_imgs.shape[1]):
            frs = []
            for fr in [test_imgs[s, t]] + [imgs[t5, 0] for t5 in top_5]:
                frs.append(fr)
                frs.append(pad)
            writer.append_data(np.concatenate(frs[:-1], 1))
        for t in range(imgs.shape[1]):
            frs = []
            for fr in [test_imgs[s, -1]] + [imgs[t5, t] for t5 in top_5]:
                frs.append(fr)
                frs.append(pad)
            writer.append_data(np.concatenate(frs[:-1], 1))
        writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Tests traj embedding model")
    parser.add_argument('model_restore', help="path to model restore checkpoint")
    parser.add_argument('test_files', help="path to folder containing new files to test against")
    parser.add_argument('--model_config', default='', type=str, help="path to model config file (including dataloader params)")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size to use during evaluation")
    parser.add_argument('--save_dir', default='embed_test', type=str, help="where to save embed test visualization")
    parser.add_argument('--loader_workers', default=4, type=int, help="number of workers for dataset loader to use")
    parser.add_argument('--mode', default='val', type=str, help="dataloader mode to use to get source files")
    parser.add_argument('--ignore_actor', default=False, type=str, help="tells source data loader to ignore particular agent")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_restore, map_location=device)
    model = model.eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    config_path = args.model_config if args.model_config else os.path.join(os.path.dirname(args.model_restore), 'config.yaml')
    config = parse_basic_config(config_path)['dataset']
    if args.ignore_actor:
        config['ignore_actor'] = args.ignore_actor
    [config.pop(c, None) for c in ('color_jitter', 'rand_gray', 'rand_crop', 'rand_rotate', 'rand_translate')]
    dataset = get_dataset(config.pop('type'))(**config, mode=args.mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_workers, drop_last=True)
    test_data = _test_files(args.test_files, config)

    args.save_dir = os.path.expanduser(args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    test_traj_embedding(model, device, loader, test_data, args.save_dir)
