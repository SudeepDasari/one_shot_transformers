from hem.robosuite.objects.custom_xml_objects import BreadObject, CerealObject, MilkObject
from hem.robosuite.objects.custom_xml_objects import BounceBox, CheezeItsBox, CleanBox, FritoBox, WooliteBox
from hem.robosuite.objects.custom_xml_objects import CokeCan, FantaCan, DrPepperCan, MDewCan, SpriteCan
from hem.robosuite.objects.custom_xml_objects import Orange, Lemon, Pear, Banana
from hem.robosuite.objects.custom_xml_objects import Whiteclaw, RedChips, PurpleChips, DutchChips
from hem.robosuite.objects.custom_xml_objects import AltoidBox, CandyBox, CardboardBox, ClaratinBox, DoveBox, FiveBox, MotrinBox, TicTacBox, ZertecBox
import numpy as np


# create train objects and names
train_objects = [BreadObject, CerealObject, MilkObject, BounceBox, CheezeItsBox, FritoBox]
train_objects.extend([WooliteBox, CokeCan, FantaCan, DrPepperCan, MDewCan, SpriteCan, Orange])
train_objects.extend([Pear, Banana, Whiteclaw, PurpleChips, DutchChips, AltoidBox, CandyBox])
train_objects.extend([CardboardBox, ClaratinBox, FiveBox, MotrinBox, TicTacBox, ZertecBox])
train_object_names = ['Bread', 'Cereal', 'Milk', 'Bounce', 'Cheezeits', 'Frito']
train_object_names.extend(['Woolitebox', 'Can', 'Fantacan', 'DrPeppercan', 'MDewcan', 'Spritecan', 'Orange'])
train_object_names.extend(['Pear', 'Banana', 'Whiteclaw', 'Purplechips', 'Dutchchips', 'Altoidbox', 'Candybox'])
train_object_names.extend(['Cardboardbox', 'Claratinbox', 'Fivebox', 'Motrinbox', 'Tictacbox', 'Zertecbox'])


# create test objects
test_objects = [Lemon, CleanBox, RedChips, DoveBox]
test_object_names = ['Lemon', 'Cleanbox', 'Redchips', 'Dovebox']


def get_train_objects(N=4):
    indices = np.random.choice(len(train_objects), size=(N,), replace=False)
    return [train_objects[i] for i in indices], [train_object_names[i] for i in indices]


def get_test_objects(N=4):
    indices = np.random.choice(len(test_objects), size=(N,), replace=False)
    return [test_objects[i] for i in indices], [train_object_names[i] for i in indices]
