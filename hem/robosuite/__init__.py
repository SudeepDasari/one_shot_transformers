def get_env(env_name):
    if env_name == 'SawyerPickPlace':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlace
        return SawyerPickPlace
    if env_name == 'SawyerPickPlaceSingle':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceSingle
        return SawyerPickPlaceSingle
    if env_name == 'SawyerPickPlaceMilk':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceMilk
        return SawyerPickPlaceMilk
    if env_name == 'SawyerPickPlaceBread':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceBread
        return SawyerPickPlaceBread
    if env_name == 'SawyerPickPlaceCereal':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceCereal
        return SawyerPickPlaceCereal
    if env_name == 'SawyerPickPlaceCan':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceCan
        return SawyerPickPlaceCan
    if env_name == 'BaxterPickPlace':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlace
        return BaxterPickPlace
    if env_name == 'BaxterPickPlaceSingle':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceSingle
        return BaxterPickPlaceSingle
    if env_name == 'BaxterPickPlaceMilk':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceMilk
        return BaxterPickPlaceMilk
    if env_name == 'BaxterPickPlaceBread':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceBread
        return BaxterPickPlaceBread
    if env_name == 'BaxterPickPlaceCereal':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceCereal
        return BaxterPickPlaceCereal
    if env_name == 'BaxterPickPlaceCan':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceCan
        return BaxterPickPlaceCan
    raise NotImplementedError


from hem.robosuite.gym_wrapper import GymWrapper
from hem.robosuite.mjc_util import postprocess_model_xml
