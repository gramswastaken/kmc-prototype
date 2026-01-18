import uuid
import os


class LogParams:
    def __init__(self, data_dir: str, name: str) -> None:
        self.name = name
        self.data_dir = data_dir
        self.log_path = log_init(self)


def log_init(params: LogParams) -> str:
    """
    Makes a directory in the data dir, it is in the format '{name}-{hash} where hash is a unique uuid'
    """
    assert " " not in params.name, "Names must not contain spaces"
    hash = str(uuid.uuid4())[:8]
    dir: str = os.path.abspath(params.data_dir) + f"/{params.name}-{hash}"

    if os.path.isdir(dir):
        print("You need to buy a lottery ticket immediately")

    os.mkdir(dir)

    return dir
