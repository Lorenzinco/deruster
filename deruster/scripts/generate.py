import random
import string
import argparse
import os
from pathlib import Path

functionnum = 0
varnum = 0

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def generate_arithmetic_operation():
    global functionnum
    global varnum
    a = random.randint(0, 100)
    b = random.randint(1, 100)
    op_code = random.randint(0, 3)

    # generate a random string
    function_name = f"function_{functionnum}"

    fuction_code = f"""
    fn {function_name}(a: i32, b: i32) -> i32 {{
        a {['+', '-', '*', '/'][op_code]} b
    }}\n
    """
    function_call = f"{function_name}({a}, {b});\n"
    functionnum += 1
    return fuction_code, function_call

def generate_variables():
    global functionnum
    global varnum
    a = random.randint(0, 100)
    b = random.randint(1, 100)

    variable_name_a = f"var_{varnum}"
    variable_name_b = f"var_{varnum+1}"

    varnum += 2


    return f"let {variable_name_a} = {a};\nlet {variable_name_b} = {b};\n"

def generate_socket_operation():
    global functionnum
    global varnum
    op_code = random.randint(0, 3)
    function_name = f"function_{functionnum}"
    function_imports = f"use std::net::UdpSocket;"
    fuction_code = f"""
    fn {function_name}() {{
        let mut buffer = [0; 10];
        let socket = UdpSocket::bind("0.0.0.0:34254").expect("couldn't bind to address");
        socket.recv_from(&mut buffer).expect("didn't receive data");
    }}\n
    """
    function_call = f"{function_name}();\n"
    functionnum += 1
    return function_imports, fuction_code, function_call


def get_project_root():
    return Path(__file__).parent.parent.parent

def generate_struct():
    global functionnum
    global varnum

    struct_name = f"struct_{varnum}"
    struct_code = f"""struct {struct_name} {{"""
    struct_declaration = f"let var_{varnum} = {struct_name} {{"
    fields = random.randint(1, 10)
    for i in range(fields):
        field_type = random.choice(["i32", "i64", "f32", "f64", "String"])
        field_name = "field_" + str(i+1)
        struct_code += f"\n    {field_name}: {field_type},"
        if field_type == "String":
            struct_declaration += f"\n    {field_name}: String::from(\"{get_random_string(10)}\"),"
        elif field_type == "i32":
            struct_declaration += f"\n    {field_name}: {random.randint(0, 100)},"
        elif field_type == "i64":
            struct_declaration += f"\n    {field_name}: {random.randint(0, 100)} as i64,"
        elif field_type == "f32":
            struct_declaration += f"\n    {field_name}: {random.uniform(0, 100)},"
        elif field_type == "f64":
            struct_declaration += f"\n    {field_name}: {random.uniform(0, 100)},"
    struct_declaration += "\n};\n"
    struct_code += "\n}\n"
    varnum += 1
    return struct_code,struct_declaration

def generate_enum():
    global functionnum
    global varnum
    enum_name = f"enum_{varnum}"
    enum_code = f"""enum {enum_name} {{"""
    enum_declaration = f"let var_{varnum} = {enum_name}::"
    variants = random.randint(1, 10)
    enum_fields = []
    for i in range(variants):
        enum_field = f"variant_{i+1}"
        enum_fields.append(enum_field)

    for field in enum_fields:
        enum_code += f"\n    {field},"
    enum_code += "\n}\n"
    enum_declaration += f"{random.choice(enum_fields)};\n"
    varnum += 1
    return enum_code,enum_declaration

def generate_dataset():
    global functionnum
    global varnum
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=int, default=1000)
    argparser.add_argument("--train", type=bool, default=True)
    args = argparser.parse_args()
    size = args.size
    for i in range(size):
        varnum = 0
        functionnum = 0
        program_imports = ""
        function_declarations = ""
        main_code = "pub fn main() {\n"
        functions = random.randint(1, 5)
        for _ in range(functions):
            function_generators = [generate_arithmetic_operation, generate_socket_operation, generate_struct, generate_enum]
            func = random.choice(function_generators)
            if func == generate_socket_operation:
                function_imports,function_code,function_call= func()
                if function_imports not in program_imports:
                    program_imports += function_imports
                main_code += function_call
                function_declarations += function_code
            else:
                function_code, function_call = func()
                main_code += function_call
                function_declarations += function_code
        main_code += "}"
        
        dataset_path = get_project_root() / "data"
        if args.train:
            dataset_path = dataset_path / "train"
        else:
            dataset_path = dataset_path / "test"
        # create the file and write the code
        os.makedirs(dataset_path/f"source", exist_ok=True)
        with open(dataset_path/f"source/{i}.rs", "w") as f:
            f.write(program_imports + "\n" + function_declarations + "\n" + main_code)



