import torch
from hem.models.imitation_module import ConditionedPolicy
from hem.models import Trainer
from hem.models.mdn_loss import GMMDistribution
import numpy as np
from hem.datasets.util import STD, MEAN, select_random_frames, resize, crop
try:
    from hem.robosuite import get_env
    from hem.robosuite.controllers.expert_pick_place import get_expert_trajectory, post_proc_obs
    mujoco_import = True
except:
    mujoco_import = False


if __name__ == '__main__':
    trainer = Trainer('imitation', "Behavior Clone model on input data conditioned on teacher video")
    config = trainer.config

    # initialize behavior cloning
    model = ConditionedPolicy(**config['policy'])
    def train_fn(m, device, context, pairs):
        images, states = pairs['images'].to(device)[:,:-1], pairs['states'].to(device)[:,:-1]
        action_dist = m({'images': images, 'states': states}, context.to(device))
        loss = -torch.mean(action_dist.log_prob(pairs['actions'].to(device)))

        stats, mean, real = {}, action_dist.mean.detach().cpu().numpy(), pairs['actions'].numpy()
        for d in range(real.shape[2]):
            stats['l_d{}'.format(d)] = np.mean(np.abs(real[:,:,d] - mean[:,:,d]))
        return loss, stats
    
    def val_fn(m, device, context, pairs):
        loss, stats = train_fn(m, device, context, pairs)
        if trainer.is_img_log_step and config.get('n_sim_rollouts', 0):
            assert mujoco_import, 'mujoco imports failed!'
            
            summary_vids, cond_frames, n_success = [], [], 0
            img_dims = (config['dataset'].get('width', 224), config['dataset'].get('height', 224))
            crop_params, norm = config['dataset'].get('crop', (0, 0, 0, 0)), config['dataset'].get('normalize', True)
            T_context, repeat = config['policy']['T_context'], config['dataset'].get('freq', 1)
            horizon = config['dataset']['T_pair']

            for _ in range(config['n_sim_rollouts']):
                rollout = [i['obs']['image'] for i in get_expert_trajectory(config['teacher_sim'])]
                cond_frame = np.concatenate((rollout[0], rollout[-1]), 1)
                cond_frames.append(resize(cond_frame, (int(img_dims[0] / 2) * 2, int(img_dims[1] / 2)), False).astype(np.float32).transpose((2, 0, 1))[None] / 255)
                
                context_images = [resize(crop(img, crop_params), img_dims, norm) for img in select_random_frames(rollout, T_context, True)]
                context_images = np.concatenate([im.transpose((2, 0, 1))[None] for im in context_images], 0)[None].astype(np.float32)
                context_images = torch.from_numpy(context_images).to(device)

                env = get_env(config['agent_sim'])(has_renderer=False, use_camera_obs=True, camera_height=320, camera_width=320)
                obs = env.reset()
                summary_vid, states, imgs, success = [obs['image'].astype(np.float32).transpose(2, 0, 1)[None] / 255], [], [], False
                obs = post_proc_obs(obs)

                for _ in range(env.horizon):
                    states = states[1:] if len(states) >= horizon else states
                    imgs = imgs[1:] if len(imgs) >= horizon else imgs
                    states.append(obs['robot-state'][None].astype(np.float32))
                    imgs.append(resize(crop(obs['image'], crop_params), img_dims, norm).transpose(2, 0, 1)[None])
                    
                    model_in = {'states': torch.from_numpy(np.concatenate(states, 0)).to(device)[None], 
                                'images': torch.from_numpy(np.concatenate(imgs, 0)).to(device)[None]}
                    action_dist = m(model_in, context_images)
                    action = action_dist.mean[0,-1].cpu().numpy()
                    for _ in range(repeat):
                        pd = np.clip(-0.4 * (obs['joint_pos'] - action[:7]), -1, 1)
                        grip = 1 if action[-1] > 0 else -1
                        a = np.concatenate((pd, [grip]))
                        obs, reward, done, _ = env.step(a)
                        success = True if reward else success
                        summary_vid.append(obs['image'].astype(np.float32).transpose(2, 0, 1)[None] / 255)
                        obs = post_proc_obs(obs)
                        if done:
                            break

                    if done or success:
                        break
                summary_vid = select_random_frames(summary_vid, 30, True)
                summary_vids.append(np.concatenate(summary_vid, 0)[None])
                n_success += int(success)

            stats['summary_vid'] = torch.from_numpy(np.concatenate(summary_vids, 0))
            stats['cond_frames'] = torch.from_numpy(np.concatenate(cond_frames, 0))
            stats['success_rate'] = n_success / config['n_sim_rollouts']
        return loss, stats
    trainer.train(model, train_fn, val_fn=val_fn)
