from hem.robosuite.objects.custom_xml_objects import BreadObject, CerealObject, MilkObject
from hem.robosuite.objects.custom_xml_objects import BounceBox, CheezeItsBox, CleanBox, FritoBox, WooliteBox
from hem.robosuite.objects.custom_xml_objects import CokeCan, FantaCan, DrPepperCan, MDewCan, SpriteCan
from hem.robosuite.objects.custom_xml_objects import Orange, Lemon, Pear, Banana
from hem.robosuite.objects.custom_xml_objects import Whiteclaw, RedChips, PurpleChips, DutchChips
from hem.robosuite.objects.custom_xml_objects import AltoidBox, CandyBox, CardboardBox, ClaratinBox, DoveBox, FiveBox, MotrinBox, TicTacBox, ZertecBox
import numpy as np


##################### FULL ENVIRONMENT SET #########################
# create train objects and names
train_objects = [BreadObject, CerealObject, MilkObject, BounceBox, CheezeItsBox, FritoBox]
train_objects.extend([WooliteBox, CokeCan, FantaCan, DrPepperCan, MDewCan, RedChips, Lemon])
train_objects.extend([Pear, Banana, Whiteclaw, PurpleChips, DutchChips, AltoidBox, CandyBox])
train_objects.extend([CardboardBox, ClaratinBox, FiveBox, MotrinBox, TicTacBox, ZertecBox])
train_object_names = ['Bread', 'Cereal', 'Milk', 'Bounce', 'Cheezeits', 'Frito']
train_object_names.extend(['Woolitebox', 'Can', 'Fantacan', 'DrPeppercan', 'MDewcan', 'Redchips', 'Lemon'])
train_object_names.extend(['Pear', 'Banana', 'Whiteclaw', 'Purplechips', 'Dutchchips', 'Altoidbox', 'Candybox'])
train_object_names.extend(['Cardboardbox', 'Claratinbox', 'Fivebox', 'Motrinbox', 'Tictacbox', 'Zertecbox'])


# create test objects
test_objects = [Orange, CleanBox, SpriteCan, DoveBox]
test_object_names = ['Orange', 'Cleanbox', 'Spritecan', 'Dovebox']


##################### PART ENVIRONMENT SET #########################
partial_train_objects = [BreadObject, CerealObject, MilkObject, CokeCan, BounceBox, MDewCan, DoveBox, Lemon, Pear, Banana]
partial_train_object_names = ['Bread', 'Cereal', 'Milk', 'Coke', 'Bounce', 'Mdew', 'Dove', 'Lemon', 'Pear', 'Banana']
partial_test_objects = [Orange, SpriteCan, AltoidBox, CleanBox]
partial_test_object_names = ['Orange', 'Sprite', 'Altoid', 'Clean']


def get_train_objects(N=4, partial=False):
    if partial:
        objs, names = partial_train_objects, partial_train_object_names
    else:
        objs, names = train_objects, train_object_names
    indices = np.random.choice(len(objs), size=(N,), replace=False)
    return [objs[i] for i in indices], [names[i] for i in indices]


def get_test_objects(N=4, partial=False):
    if partial:
        objs, names = partial_test_objects, partial_test_object_names
    else:
        objs, names = test_objects, test_object_names
    indices = np.random.choice(len(objs), size=(N,), replace=False)
    return [objs[i] for i in indices], [names[i] for i in indices]
