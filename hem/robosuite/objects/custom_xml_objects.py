import os
import hem.robosuite.objects
from robosuite.models.objects import MujocoXMLObject


BASE_DIR = os.path.join(os.path.dirname(__file__), 'xml')


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


class CokeCan(MujocoXMLObject):
    """
    Coke can object (used in SawyerPickPlace)
    """

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'coke.xml'))
CanObject = CokeCan


class SpriteCan(MujocoXMLObject):

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'sprite.xml'))


class MDewCan(MujocoXMLObject):

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'mdew.xml'))


class FantaCan(MujocoXMLObject):

    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'fanta.xml'))


class DrPepperCan(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'drpepper.xml'))


class CheezeItsBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'cheezeits.xml'))


class FritoBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'frito.xml'))


class BounceBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'bounce.xml'))


class CleanBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'clean.xml'))


class WooliteBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'woolite.xml'))


class Lemon(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'lemon.xml'))


class Pear(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'pear.xml'))


class Banana(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'banana.xml'))


class Orange(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'orange.xml'))


class RedChips(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'redchips.xml'))


class PurpleChips(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'purplechips.xml'))


class DutchChips(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'dutchchips.xml'))


class Whiteclaw(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'whiteclaw.xml'))


class AltoidBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'altoid.xml'))


class CandyBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'candy.xml'))


class CardboardBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'cardboard.xml'))


class ClaratinBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'claratin.xml'))


class DoveBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'dove.xml'))


class FiveBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'five.xml'))


class MotrinBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'motrin.xml'))


class TicTacBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'tictac.xml'))


class ZertecBox(MujocoXMLObject):
    def __init__(self):
        super().__init__(os.path.join(BASE_DIR, 'zertec.xml'))
