try:
    from mujoco_py import load_model_from_xml, MjSim, MjRenderContextOffscreen
    from hem.robosuite import postprocess_model_xml
except:
    # in case experiments don't require rendering ignore failure
    pass


class ImageRenderWrapper:
    def __init__(self, traj, height=320, width=320, depth=False, no_render=False):
        self._height = height
        self._width = width
        self._sim = None
        self._traj = traj
        self._depth = depth
        self._no_render = no_render

    def get(self, t, decompress=True):
        ret = self._traj[t]
        if decompress and 'image' not in ret['obs'] and not self._no_render:
            sim = self._get_sim()
            sim.set_state_from_flattened(self._traj.get_raw_state(t))
            sim.forward()
            if self._depth:
                image, depth = sim.render(camera_name='frontview', width=self._width, height=self._height, depth=True)
                ret['obs']['image'] = image[:,::-1]
                ret['obs']['depth'] = self._proc_depth(depth[:,::-1])
            else:
                ret['obs']['image'] = sim.render(camera_name='frontview', width=self._width, height=self._height, depth=False)[80:,::-1]
        return ret

    def _proc_depth(self, depth):
        if self._depth_norm == 'sawyer':
            return (depth - 0.992) / 0.0072
        return depth
    
    def _get_sim(self):
        if self._sim is not None:
            return self._sim

        xml = postprocess_model_xml(self._traj.config_str)
        self._depth_norm = None
        if 'sawyer' in xml:
            from hem.datasets.precompiled_models.sawyer import models
            self._sim = models[0]
            self._depth_norm = 'sawyer'
        elif 'baxter' in xml:
            from hem.datasets.precompiled_models.baxter import models
            self._sim = models[0]
        elif 'panda' in xml:
            from hem.datasets.precompiled_models.panda import models
            self._sim = models[0]
        else:
            model = load_model_from_xml(xml)
            model.vis.quality.offsamples = 8
            sim = MjSim(load_model_from_xml(xml))
            render_context = MjRenderContextOffscreen(sim)
            render_context.vopt.geomgroup[0] = 0
            render_context.vopt.geomgroup[1] = 1 
            sim.add_render_context(render_context)
            self._sim = sim

        return self._sim

    def __getitem__(self, t):
        return self.get(t)

    def __len__(self):
        return len(self._traj)
    
    def __iter__(self):
        for d in range(len(self._traj)):
            yield self.get(d)

    @property
    def config_str(self):
        return self._traj.config_str
