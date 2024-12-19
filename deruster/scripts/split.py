import re
from pathlib import Path
from argparse import ArgumentParser
import os
import tree_sitter_rust
from tree_sitter import Language, Parser

def get_project_root():
    return Path(__file__).parent.parent.parent

def extract_functions(node,functions):
    if node.type == "function_item":
        functions.append(node.text)
    for child in node.children:
        extract_functions(child,functions)

def split_functions(source_files,sources_path):
    RUST_LANGUAGE = Language(tree_sitter_rust.language())
    parser = Parser(RUST_LANGUAGE)

    for source in source_files:
        file_path = sources_path / source
        with open(file_path, "r") as f:
            source_code = f.read()
        tree = parser.parse(bytes(source_code, "utf8"))
        functions = []
        extract_functions(tree.root_node,functions)
        #create a folder for each source file
        source_folder = sources_path / source.split(".")[0]
        os.makedirs(source_folder,exist_ok=True)
        for i,function in enumerate(functions):
            if i == len(functions) - 1:
                function_file = source_folder / f"main.rs"
            else:
                function_file = source_folder / f"function_{i}.rs"
            with open(function_file,"wb") as f:
                f.write(function)


def split_assembly(assembly_files, assemblies_path):
    for assembly in assembly_files:
        functions = []
        assembly_path = assemblies_path / assembly
        with open(assembly_path, "r") as f:
            assembly_code = f.read()

        function_regex = r"(__.*::function_[0-9]+:\n)"
        main_regex = r"(__.*::main:\n)"
        # split on the main function
        main = re.split(main_regex, assembly_code)
        if main:
            # main code is main[1] + main[2]
            functions.append(main[1] + main[2])

        parts = re.split(function_regex, main[0])

        for i in range(1, len(parts), 2):
            functions.append((parts[i], parts[i + 1]))

        assembly_folder = assemblies_path / assembly.split(".")[0]


        os.makedirs(assembly_folder,exist_ok=True)
        function_file = assembly_folder / f"main.s"

        with open(function_file,"w") as f:
            f.write(functions[0])
        
        for function in functions[1:]:
            function_number = function[0].split("_")[-1]
            #remove : from the function number
            function_number = function_number[:-1]
            function_file = assembly_folder / f"function_{function_number}.s"
            with open(function_file,"w") as f:
                f.write(function[0]+function[1])



    
def split():
    print("Splitting...")
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
    source_files = [file for file in source_files if os.path.isfile(sources_path / file)]
    assembly_files = os.listdir(assemblies_path)
    assembly_files = [file for file in assembly_files if os.path.isfile(assemblies_path / file)]

    split_functions(source_files,sources_path)
    split_assembly(assembly_files,assemblies_path)
    


    #file_name = str(file_index)
    #function_regex = f"__{file_name}::function_([0-9]+):\n"

if __name__ == "__main__":
    split()