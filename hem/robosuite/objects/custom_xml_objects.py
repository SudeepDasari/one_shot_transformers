import os
import hem.robosuite.objects
from robosuite.models.objects import MujocoXMLObject


BASE_DIR = os.path.join(os.path.dirname(hem.robosuite.objects.__file__), 'xml')


class CerealObject(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'cereal.xml'))


class BreadObject(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'bread.xml'))


class MilkObject(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'milk.xml'))


class CanObject(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'can.xml'))
