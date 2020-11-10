from robosuite.models.tasks import PickPlaceTask as DefaultPickPlaceTask
from robosuite.utils import RandomizationError
from robosuite.utils.mjcf_utils import array_to_string
import numpy as np


class PickPlaceTask(DefaultPickPlaceTask):
    def place_objects(self):
        """Places objects randomly until no collisions or max iterations hit."""
        placed_objects = []
        index = 0

        # place objects by rejection sampling
        for _, obj_mjcf in self.mujoco_objects.items():
            horizontal_radius = obj_mjcf.get_horizontal_radius()
            bottom_offset = obj_mjcf.get_bottom_offset()
            success = False
            for _ in range(5000):  # 5000 retries
                bin_x_half = self.bin_size[0] / 2 - horizontal_radius - 0.01
                bin_y_half = self.bin_size[1] / 2 - horizontal_radius - 0.03
                object_x = np.random.uniform(high=bin_x_half, low=-bin_x_half)
                object_y = np.random.uniform(high=bin_y_half, low=-bin_y_half)

                # make sure objects do not overlap
                object_xy = np.array([object_x, object_y, 0])
                pos = self.bin_offset - bottom_offset + object_xy
                location_valid = True
                for pos2, r in placed_objects:
                    dist = np.linalg.norm(pos[:2] - pos2[:2], np.inf)
                    if dist <= r + horizontal_radius:
                        location_valid = False
                        break

                # place the object
                if location_valid:
                    # add object to the position
                    placed_objects.append((pos, horizontal_radius))
                    self.objects[index].set("pos", array_to_string(pos))
                    # random z-rotation
                    quat = self.sample_quat(obj_mjcf.name)
                    self.objects[index].set("quat", array_to_string(quat))
                    success = True
                    break

            # raise error if all objects cannot be placed after maximum retries
            if not success:
                raise RandomizationError("Cannot place all objects in the bins")
            index += 1

    def sample_quat(self, obj_name):
        """Samples quaternions of random rotations along the z-axis."""
        if self.z_rotation:
            rot_angle = np.random.uniform(high=np.pi/4, low=0) if np.random.uniform() < 0.5 else np.random.uniform(low=3*np.pi/4, high=2*np.pi)
            if obj_name in ('pear', 'banana'):
                rot_angle = np.random.uniform(high=np.pi/4, low=-np.pi/4)
            return [np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]
        return [1, 0, 0, 0]
