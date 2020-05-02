from hem.datasets.savers.trajectory import Trajectory
from hem.datasets.savers.hdf5_trajectory import HDF5Trajectory
import pickle as pkl
import glob
import os


def get_dataset(name):
    if name == 'something something':
        from .something_dataset import SomethingSomething
        return SomethingSomething
    elif name == 'agent teacher':
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
    elif name == 'paired frames':
        from .frame_datasets import PairedFrameDataset
        return PairedFrameDataset
    elif name == 'unpaired frames':
        from .frame_datasets import UnpairedFrameDataset
        return UnpairedFrameDataset
    raise NotImplementedError


def get_validation_batch(loader, batch_size=8):
    pass


def load_traj(fname):
    if '.pkl' in fname:
        return pkl.load(open(fname, 'rb'))['traj']
    elif '.hdf5' in fname:
        traj = HDF5Trajectory()
        traj.load(fname)
        return traj
    raise NotImplementedError


def get_files(root_dir):
    root_dir = os.path.expanduser(root_dir)
    if 'pkl' in root_dir or 'hdf5' in root_dir:
        return sorted(glob.glob(root_dir))
    pkl_files = glob.glob(root_dir + '*.pkl')
    hdf5_files = glob.glob(root_dir + '*.hdf5')
    return sorted(pkl_files + hdf5_files)
