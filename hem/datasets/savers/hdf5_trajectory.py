from hem.datasets.savers.trajectory import _compress_obs, _decompress_obs, Trajectory
import copy
import h5py


class _HDF5BackedData:
    def __init__(self, hf, length):
        self._hf = hf
        self._len = length
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        group_t = self._hf[str(index)]
        obs_t = {}
        for k in group_t['obs'].keys():
            obs_t[k] = group_t['obs'][k][:]
        reward_t = group_t.get('reward', None)
        done_t = group_t.get('done', None)
        info_t = group_t.get('info', None)
        action_t = None
        if 'action' in group_t:
            action_t=group_t.get('action')[:]
        return obs_t, reward_t, done_t, info_t, action_t


class HDF5Trajectory(Trajectory):
    def __init__(self, fname=None, traj=None, config_str=None):
        self._hf_name = None
        if traj is not None:
            assert fname is None
            self.set_config_str(traj.config_str)
            self._data = copy.deepcopy(traj._data)
            self._raw_state = copy.deepcopy(traj._raw_state)
        elif fname is not None:
            self.load(fname)
        else:
            super().__init__(config_str)

    def append(self, obs, reward=None, done=None, info=None, action=None, raw_state=None):
        raise NotImplementedError("Cannot Append to HDF5 backed trajectory!")

    def load(self, fname):
        hf = h5py.File(fname, 'r')
        self._config_str = hf.get('config_str', None)
        self._raw_state = hf.get('raw_state', None)

        cntr = 0
        while str(cntr) in hf:
            cntr += 1
        self._data = _HDF5BackedData(hf, cntr)
        if self._raw_state is None:
            self._raw_state = [None for _ in range(cntr)]
    
    def to_pkl_traj(self):
        traj = Trajectory()
        traj._config_str = copy.deepcopy(self._config_str)
        traj._raw_state = copy.deepcopy(self._raw_state)
        traj._data = [self._data[t] for t in range(len(self._data))]
        return traj

    def save(self, fname):
        with h5py.File(fname, 'w') as hf:
            if self._config_str:
                hf.create_dataset('config_str', data=self._config_str)
            if any(self._raw_state):
                hf.create_dataset('raw_state', data=self._raw_state)

            cntr = 0
            for obs_t, reward_t, done_t, info_t, action_t in self._data:
                group_t = hf.create_group(str(cntr))
                if obs_t:
                    obs_group = group_t.create_group('obs')
                    for k, v in obs_t.items():
                        obs_group.create_dataset(k, data=v)
                if reward_t is not None:
                    group_t.create_dataset('reward', data=reward_t)
                if done_t is not None:
                    group_t.create_dataset('done', data=done_t)
                if info_t is not None:
                    group_t.create_dataset('info', data=info_t)
                if action_t is not None:
                    group_t.create_dataset('action', data=action_t)
                cntr += 1
