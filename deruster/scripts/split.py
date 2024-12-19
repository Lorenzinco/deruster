import re
from pathlib import Path
from argparse import ArgumentParser
import os

def get_project_root():
    return Path(__file__).parent.parent.parent

def split_functions(source_files,sources_path):
    for source in source_files:
        file_path = sources_path / source
        with open(file_path, "r") as f:
            source_code = f.read()
        function_regex = r"fn ([a-zA-Z0-9_]+)\("
        functions = re.findall(function_regex, source_code)
        functions_path = sources_path / source.split(".")[0]
        os.makedirs(functions_path, exist_ok=True)
        for function in functions:
            function_code = f"fn {function}("
            function_code += re.search(f"fn {function}\((.|\n)+", source_code).group(1)
            function_code += "\n"
            function_file = functions_path / f"{function}.rs"
            with open(function_file, "w") as f:
                f.write(function_code)
                exit(0)


    
def split():
    print("Splitting")
    args = ArgumentParser()
    args.add_argument("--train", action="store_true")
    args = args.parse_args()
    project_root = get_project_root()
    data_path = project_root / "data"
    if args.train:
        dataset_path = data_path / "train"
    else:
        dataset_path = data_path / "test"
    sources_path = dataset_path / "source"
    assemblies_path = dataset_path / "assembly"

    source_files = os.listdir(sources_path)
    assembly_files = os.listdir(assemblies_path)

    split_functions(source_files,sources_path)
    #split_assembly(assembly_files)
    


    #file_name = str(file_index)
    #function_regex = f"__{file_name}::function_([0-9]+):\n"
