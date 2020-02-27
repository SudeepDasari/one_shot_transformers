from mujoco_py import load_model_from_xml, MjSim, MjRenderContextOffscreen
from hem.robosuite import postprocess_model_xml


def create_model(xml):
    sim = MjSim(load_model_from_xml(postprocess_model_xml(xml)))
    render_context = MjRenderContextOffscreen(sim)
    render_context.vopt.geomgroup[0] = 0
    render_context.vopt.geomgroup[1] = 1 
    sim.add_render_context(render_context)
    return sim
