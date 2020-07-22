from hem.robosuite.sawyer.sawyer_pick_place import SawyerPickPlace
from robosuite.environments.sawyer import SawyerEnv
from hem.robosuite.objects import get_train_objects, get_test_objects
from hem.robosuite.tasks.pick_place import PickPlaceTask
from collections import OrderedDict
from robosuite.utils.mjcf_utils import string_to_array
from hem.robosuite.arena.bin_arena import BinsArenaNoWall
import numpy as np


class SawyerPickDiverseObj(SawyerPickPlace):
    def __init__(self, train_objects, **kwargs):
        self.ob_inits, self.item_names = get_train_objects() if train_objects else get_test_objects()
        super().__init__(**kwargs)

    def _load_model(self):
        SawyerEnv._load_model(self)
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = BinsArenaNoWall(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )

        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([.5, -0.3, 0])
        self.item_names_org = list(self.item_names)
        self.obj_to_use = (self.item_names[0] + "{}").format(0)

        lst = []
        for i in range(len(self.ob_inits)):
            ob = self.ob_inits[i]()
            lst.append((str(self.item_names[i]) + "0", ob))

        self.mujoco_objects = OrderedDict(lst)
        self.n_objects = len(self.mujoco_objects)

        # task includes arena, robot, and objects of interest
        self.model = PickPlaceTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            []
        )
        self.model.place_objects()
        self.model.place_visual()
        self.bin_pos = string_to_array(self.model.bin2_body.get("pos"))
        self.bin_size = self.model.bin_size
    

class SawyerPickPlaceDiverseTrain(SawyerPickDiverseObj):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, randomize_goal=True, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        obj = np.random.choice(items) if force_object is None else force_object
        obj = items[obj] if isinstance(obj, int) else obj
        super().__init__(train_objects=True, single_object_mode=2, object_type=obj, no_clear=True, randomize_goal=randomize_goal, **kwargs)


class SawyerPickPlaceDiverseTest(SawyerPickDiverseObj):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, force_object=None, randomize_goal=True, **kwargs):
        assert "single_object_mode" not in kwargs, "invalid set of arguments"
        items = ['milk', 'bread', 'cereal', 'can']
        obj = np.random.choice(items) if force_object is None else force_object
        obj = items[obj] if isinstance(obj, int) else obj
        super().__init__(train_objects=False, single_object_mode=2, object_type=obj, no_clear=True, randomize_goal=randomize_goal, **kwargs)
