from hem.datasets.precompiled_models.util import create_model


source_folder = '/'.join(__file__.split('/')[:-1]) + '/sawyer/'

# base model
base_model = create_model(open(source_folder + 'base.xml', 'r').read())

# all models
models = [base_model]
