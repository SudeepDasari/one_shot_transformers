from hem.datasets.agent_dataset import AgentDemonstrations
from hem.datasets.util import resize
import random


class TeacherDemonstrations(AgentDemonstrations):
    def proc_traj(self, traj):
        return self._make_context(traj)


if __name__ == '__main__':
    import time
    import imageio
    from torch.utils.data import DataLoader
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--N', default=1)
    parser.add_argument('--T', default=15)
    parser.add_argument('--B', default=10)
    args = parser.parse_args()

    ag = TeacherDemonstrations(args.path, normalize=False, T_context=args.T)
    loader = DataLoader(ag, batch_size = args.B, num_workers=8)

    start = time.time()
    timings = []
    for context in loader:
        timings.append(time.time() - start)
        print(context.shape)

        if len(timings) > args.N:
            break
        start = time.time()
    print('avg ex time', sum(timings) / len(timings) / args.B)

    out = imageio.get_writer('out1.gif')
    for t in range(context.shape[1]):
        frame = [np.transpose(fr, (1, 2, 0)) for fr in context[:, t]]
        frame = np.concatenate(frame, 1)
        out.append_data(frame.astype(np.uint8))
    out.close()
