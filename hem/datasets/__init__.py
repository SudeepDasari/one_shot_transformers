def get_dataset(name):
    if name == 'something something':
        from .something_dataset import SomethingSomething
        return SomethingSomething
    raise NotImplementedError


def get_validation_batch(loader, batch_size=8):
    pass
