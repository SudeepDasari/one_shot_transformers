from collections import OrderedDict
import cv2
import copy


class Trajectory:
    def __init__(self):
        self._data = []
    
    def add(self, obs, reward, done, info):
        if 'image' in obs:
            okay, im_string = cv2.imencode('.jpg', obs['image'])
            assert okay, "image encoding failed!"
            obs['image'] = im_string
        self._data.append(copy.deepcopy((obs, reward, done, info)))

    @property
    def T(self):
        return len(self._data)
    
    def __getitem__(self, t):
        return self.get(t)

    def get(self, t):
        assert isinstance(t, int), "t should be an integer value!"
        assert 0 <= t < self.T, "index should be in [0, T)"

        data_t = copy.deepcopy(self._data[t])
        if 'image' in data_t[0]:
            data_t[0]['image'] = cv2.imdecode(data_t[0]['image'], cv2.IMREAD_COLOR)
        return data_t

    def __len__(self):
        return self.T

    def __iter__(self):
        for d in range(self.T):
            yield self.get(d)
