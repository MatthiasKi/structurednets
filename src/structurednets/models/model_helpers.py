from structurednets.models.visionmodel import VisionModel
from structurednets.models.alexnet import AlexNet
from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.models.vgg16 import VGG16

def visionmodel_name_to_class(name: str) -> VisionModel:
    if name == AlexNet.__name__:
        return AlexNet
    elif name == GoogleNet.__name__:
        return GoogleNet
    elif name == InceptionV3.__name__:
        return InceptionV3
    elif name == MobilenetV2.__name__:
        return MobilenetV2
    elif name == Resnet18.__name__:
        return Resnet18
    elif name == VGG16.__name__:
        return VGG16
    else:
        raise Exception("Visionmodel Name not recognized: " + name)