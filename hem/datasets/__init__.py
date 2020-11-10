from hem.datasets.savers.trajectory import Trajectory
from hem.datasets.savers.hdf5_trajectory import HDF5Trajectory
import pickle as pkl
import glob
import os
import re
try:
    raise NotImplementedError
    from hem.datasets.savers.render_loader import ImageRenderWrapper
    import_render_wrapper = True
except:
    import_render_wrapper = False


def get_dataset(name):
    if name == 'agent teacher':
        from .agent_teacher_dataset import AgentTeacherDataset
        return AgentTeacherDataset
    elif name == 'paired agent teacher':
        from .agent_teacher_dataset import PairedAgentTeacherDataset
        return PairedAgentTeacherDataset
    elif name == 'labeled agent teacher':
        from .agent_teacher_dataset import LabeledAgentTeacherDataset
        return LabeledAgentTeacherDataset
    elif name == 'agent':
        from .agent_dataset import AgentDemonstrations
        return AgentDemonstrations
    raise NotImplementedError


def get_validation_batch(loader, batch_size=8):
    pass


def load_traj(fname):
    if '.pkl' in fname:
        traj = pkl.load(open(fname, 'rb'))['traj']
    elif '.hdf5' in fname:
        traj = HDF5Trajectory()
        traj.load(fname)
    else:
        raise NotImplementedError

    traj = traj if not import_render_wrapper else ImageRenderWrapper(traj)
    return traj


def get_files(root_dir):
    def natural_keys(text):
        # source: https://www.tutorialspoint.com/How-to-correctly-sort-a-string-with-a-number-inside-in-Python
        atoi = lambda text: int(text) if text.isdigit() else text
        return [atoi(c) for c in re.split('(\d+)',text)]

    root_dir = os.path.expanduser(root_dir)
    if 'pkl' in root_dir or 'hdf5' in root_dir:
        return sorted(glob.glob(root_dir), key=natural_keys)
    pkl_files = glob.glob(root_dir + '*.pkl')
    hdf5_files = glob.glob(root_dir + '*.hdf5')
    return sorted(pkl_files + hdf5_files, key=natural_keys)
