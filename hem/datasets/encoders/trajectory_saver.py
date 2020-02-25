from collections import OrderedDict
import cv2
import copy


def _compress_obs(obs):
    if 'image' in obs:
        okay, im_string = cv2.imencode('.jpg', obs['image'])
        assert okay, "image encoding failed!"
        obs['image'] = im_string
    return obs


def _decompress_obs(obs):
    if 'image' in obs:
        obs['image'] = cv2.imdecode(obs['image'], cv2.IMREAD_COLOR)
    return obs


class Trajectory:
    def __init__(self, sim_xml=None):
        self._data = []
        self._mj_state = []
        self._actions = []
        self._final_obs = None
        self._sim_xml = sim_xml
    
    def add(self, obs, reward, done, info, action=None, mj_state=None):
        obs = _compress_obs(obs)
        self._data.append(copy.deepcopy((obs, reward, done, info)))
        self._actions.append(action)
        self._mj_state.append(copy.deepcopy(mj_state))

    def log_final(self, obs, mj_state=None):
        self._final_obs = _compress_obs(obs)
        self._mj_state.append(copy.deepcopy(mj_state))

    @property
    def T(self):
        return len(self._data)
    
    def __getitem__(self, t):
        return self.get(t)

    def get(self, t):
        assert isinstance(t, int), "t should be an integer value!"
        assert 0 <= t < self.T + 1, "index should be in [0, T]"

        if t == self.T:
            obs_t = _decompress_obs(copy.deepcopy(self._final_obs))
            return dict(obs=obs_t)
        
        obs_t, reward_t, done_t, info_t = copy.deepcopy(self._data[t])
        obs_t = _decompress_obs(obs_t)
        return dict(obs=obs_t, reward=reward_t, done=done_t, info=info_t, action=self._actions[t])

    def __len__(self):
        return self.T

    def __iter__(self):
        for d in range(self.T + 1):
            yield self.get(d)

    def get_sim_state(self, t):
        assert 0 <= t < self.T + 1, "index should be in [0, T]"
        return copy.deepcopy(self._mj_state[t])

    def log_xml(self, xml_str):
        self._sim_xml = xml_str

    @property
    def sim_xml(self):
        return self._sim_xml
