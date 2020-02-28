def get_model(name):
    if name == 'base':
        from .basic_embedding import BasicEmbeddingModel
        return BasicEmbeddingModel
    elif name == 'base rnn':
        from .basic_rnn import BaseRNN
        return BaseRNN
    raise NotImplementedError
