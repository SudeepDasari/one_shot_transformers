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
    def __init__(self, config_str=None):
        self._data = []
        self._raw_state = []
        self.set_config_str(config_str)
    
    def append(self, obs, reward=None, done=None, info=None, action=None, raw_state=None):
        """
        Logs observation and rewards taken by environment as well as action taken
        """
        obs, reward, done, info, action, raw_state = [copy.deepcopy(x) for x in [obs, reward, done, info, action, raw_state]]

        obs = _compress_obs(obs)
        self._data.append((obs, reward, done, info, action))
        self._raw_state.append(raw_state)

    @property
    def T(self):
        """
        Returns number of states
        """
        return len(self._data)
    
    def __getitem__(self, t):
        return self.get(t)

    def get(self, t):
        assert isinstance(t, int), "t should be an integer value!"
        assert 0 <= t < self.T, "index should be in [0, T)"
        
        obs_t, reward_t, done_t, info_t, action_t = copy.deepcopy(self._data[t])
        obs_t = _decompress_obs(obs_t)
        ret_dict = dict(obs=obs_t, reward=reward_t, done=done_t, info=info_t, action=action_t)

        for k in list(ret_dict.keys()):
            if ret_dict[k] is None:
                ret_dict.pop(k)
        return ret_dict

    def __len__(self):
        return self.T

    def __iter__(self):
        for d in range(self.T):
            yield self.get(d)

    def get_raw_state(self, t):
        assert 0 <= t < self.T, "index should be in [0, T)"
        return copy.deepcopy(self._raw_state[t])

    def set_config_str(self, config_str):
        self._config_str = config_str

    @property
    def config_str(self):
        return self._config_str
