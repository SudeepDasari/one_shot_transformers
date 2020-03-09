from hem.models.trainer import Trainer


def get_model(name):
    if name == 'base':
        from .basic_embedding import BasicEmbeddingModel
        return BasicEmbeddingModel
    elif name == 'base rnn':
        from .basic_rnn import BaseRNN
        return BaseRNN
    elif name == 'resnet':
        from .basic_embedding import ResNetFeats
        return ResNetFeats
    elif name == 'vgg':
        from .basic_embedding import VGGFeats
        return VGGFeats
    raise NotImplementedError
