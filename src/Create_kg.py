### ~~~ GLOBAL IMPORTS ~~~ ###
from dataclasses import dataclass


### ~~~ LOCAL IMPORTS ~~~ ###
# None


### ~~~ TYPE DEFINITIONS ~~~ ###
@dataclass
class Config:
    uri: str
    username: str
    password: str


### ~~~ STATE MANAGEMENT ~~~ ###
# None


def load_config(path: str = "./config.txt") -> Config:
    """"""
    ### read config file ###
    with open(path, "r") as file:
        lines = file.readlines()

    ### parse config file ###
    config_dict = {}
    for line in lines:
        key, value = line.strip().split("=")
        config_dict[key] = value

    ### insure all keys are present ###
    required_keys = {"uri", "username", "password"}
    if not required_keys.issubset(config_dict.keys()):
        missing_keys = required_keys - config_dict.keys()
        raise KeyError(f"Missing keys in config file: {missing_keys}")

    ### create Config instance ###
    config: Config = Config(
        uri=config_dict["uri"],
        username=config_dict["username"],
        password=config_dict["password"],
    )

    return config


def main() -> int:
    """ """
    return 0


if __name__ == "__main__":
    exit(main())
