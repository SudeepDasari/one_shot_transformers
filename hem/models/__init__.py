def get_model(name):
    if name == 'base':
        from .basic_embedding import BasicEmbeddingModel
        return BasicEmbeddingModel
    raise NotImplementedError
