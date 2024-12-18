import os
from pathlib import Path
from argparse import ArgumentParser


def get_project_root():
    return Path(__file__).parent.parent.parent

def compile():
    args = ArgumentParser()
    args.add_argument("--demangle", action="store_true")
    args = args.parse_args()
    script_path = get_project_root() / "data"/"train"/"compile.sh"

    if args.demangle:
        os.system(f"sh {script_path} --demangle")
    else:
        os.system(f"sh {script_path} --compile")
    