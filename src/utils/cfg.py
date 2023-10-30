from typing import Any


class Config(dict):
    def get_(self, key: str, default: Any = None):
        o: Any = self
        for item in key.split("."):
            o = o.get(item, default)
        return o


def _get_config() -> Config:
    while True:
        try:
            import tomllib

            with open("config.toml", "rb") as f:
                return Config(tomllib.load(f))
        except Exception as e:
            raise Exception("No toml support!") from e


config = _get_config()
