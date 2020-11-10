import re, yaml, os, json


def parse_basic_config(config_file, resolve_env=True):
    """
    Configures custom yaml loading behavior and parses config file
    """
    if resolve_env:
        env_pattern = re.compile(r"\$\{(.*?)\}(.*)", re.VERBOSE)
        def env_var_constructor(loader, node):
            """
            Converts ${VAR}/* from config file to 'os.environ[VAR] + *'
            Modified from: https://www.programcreek.com/python/example/61563/yaml.add_implicit_resolver
            """
            value = loader.construct_scalar(node)
            env_var, remainder = env_pattern.match(value).groups()
            if env_var not in os.environ:
                raise ValueError("config requires envirnonment variable {} which is not set".format(env_var))
            return os.environ[env_var] + remainder
        yaml.add_implicit_resolver("!env", env_pattern, Loader=yaml.SafeLoader)
        yaml.add_constructor('!env', env_var_constructor, Loader=yaml.SafeLoader)

    with open(os.path.expanduser(config_file), 'r') as config:
        return yaml.load(config, Loader=yaml.SafeLoader)


def clean_dict(state_dict):
    for k in list(state_dict.keys()):
        if 'module' in k:
            state_dict[k[7:]] = state_dict.pop(k)
    return state_dict


def get_kl_beta(config, step):
    beta = config['kl_beta']
    step = step % config['kl_cycle'] if 'kl_cycle' in config and step < config.get('max_kl_cycle', float('inf')) else step
    assert beta >=0
    if 'kl_anneal' in config:
        beta_0, start, end = config['kl_anneal']
        assert end > start and beta_0 >= 0
        alpha = min(max(float(step - start) / (end - start), 0), 1)
        beta = (1 - alpha) * beta_0 + alpha * beta
    return beta
