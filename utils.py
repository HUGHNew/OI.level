import os


def file_exists(file: str) -> bool:
    return os.path.exists(file) and os.path.isfile(file)

def apply(_l: list, f, isMutable:bool = True):
    for idx in range(len(_l)):
        if isMutable: f(_l[idx])
        else: _l[idx] = f(_l[idx])