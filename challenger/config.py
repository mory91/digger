import os
from dataclasses import dataclass, fields


@dataclass
class Global:
    _name: str = 'glob'
    GPU: int = 0


class _Config(object):
    _instance = None
    configs_list = [
        Global,
    ]

    def create_configs(self):
        for config in _Config.configs_list:
            name = config._name
            config_class = config()
            for field in fields(config_class):
                default_value = getattr(config_class, field.name)
                value = os.environ.get(
                    field.name,
                    default=default_value
                )
                value = field.type(value)
                setattr(config_class, field.name, value)
            setattr(self, name, config_class)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_Config, cls).__new__(cls)
            cls._instance.create_configs()
        return cls._instance


def get_config():
    return _Config()
