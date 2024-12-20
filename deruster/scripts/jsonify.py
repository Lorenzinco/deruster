import json
from pathlib import Path
import os

def get_project_root():
    return Path(__file__).parent.parent.parent

def jsonify_functions(source_files,sources_path,assemby_files,assemblies_path):
    functions = []
    for source in source_files:
        source_folder = sources_path / source.split(".")[0]
        assembly_folder = assemblies_path / source.split(".")[0]
        for function_file in source_folder.iterdir():
            function_assembly_file = assembly_folder / function_file.name.replace(".rs", ".s")
            with open(function_file, "r") as f:
                function_code = f.read()
            try:
                with open(assemblies_path / function_assembly_file, "r") as f:
                    assembly_code = f.read()
            except FileNotFoundError:
                continue
            functions.append({"instruction": assembly_code, "output": function_code})
    return functions

if __name__ == "__main__":
    project_root = get_project_root()
    sources_path = project_root / "data" / "train" / "source"
    assembly_path = project_root / "data" / "train" / "assembly"

    source_files = os.listdir(sources_path)
    assembly_files = os.listdir(assembly_path)

    functions = jsonify_functions(source_files, sources_path, assembly_files, assembly_path)
    json_path = project_root / "data" / "rust-decompile.json"
    with open(json_path, "w") as f:
        json.dump(functions, f, indent=4)