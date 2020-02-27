try:
    from mujoco_py import load_model_from_xml, MjSim, MjRenderContextOffscreen
    from hem.robosuite import postprocess_model_xml
except:
    # in case experiments don't require rendering ignore failure
    pass


class ImageRenderWrapper:
    def __init__(self, traj, height=None, width=None, no_render=False):
        self._height = height
        self._width = width
        self._sim = None
        self._traj = traj
        self._no_render = no_render
    
    def get(self, t):
        ret = self._traj[t]
        if 'image' not in ret['obs'] and not self._no_render:
            sim = self._get_sim()
            sim.set_state_from_flattened(self._traj.get_raw_state(t))
            sim.forward()
            ret['obs']['image'] = sim.render(camera_name='frontview', width=self._width, height=self._height, depth=False)[:,::-1]
        return ret
    
    def _get_sim(self):
        if self._sim is not None:
            return self._sim

        xml = postprocess_model_xml(self._traj.config_str)
        if 'sawyer' in xml and 'can' in xml:
            from hem.datasets.precompiled_models.sawyer_can import models
            self._sim = models[0]
        elif 'baxter' in xml and 'can' in xml:
            from hem.datasets.precompiled_models.baxter_can import models
            self._sim = models[0]
        else:
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
