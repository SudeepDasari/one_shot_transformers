def get_env(env_name):
    if env_name == 'SawyerPickPlaceDistractor':
        from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlaceDistractor
        return SawyerPickPlaceDistractor
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
    if env_name == 'BaxterPickPlaceDistractor':
        from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceDistractor
        return BaxterPickPlaceDistractor
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
    if env_name == 'PandaPickPlaceDistractor':
        from hem.robosuite.panda.panda_pick_place import PandaPickPlaceDistractor
        return PandaPickPlaceDistractor
    raise NotImplementedError


from hem.robosuite.gym_wrapper import GymWrapper
from hem.robosuite.mjc_util import postprocess_model_xml
