import os
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.parent

def compile():
    script_path = get_project_root() / "data"/"train"/"compile.sh"
    os.system(f"sh {script_path}")
    