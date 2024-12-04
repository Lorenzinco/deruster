import random
import string
import argparse
import os
from pathlib import Path

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def generate_arithmetic_operation():
    a = random.randint(0, 100)
    b = random.randint(1, 100)
    op_code = random.randint(0, 3)

    # generate a random string
    function_name = get_random_string(10)

    fuction_code = f"""
    fn {function_name}(a: i32, b: i32) -> i32 {{
        a {['+', '-', '*', '/'][op_code]} b
    }}\n
    """
    function_call = f"{function_name}({a}, {b});\n"
    return fuction_code, function_call

def generate_variables():
    a = random.randint(0, 100)
    b = random.randint(1, 100)
    variable_name_a = get_random_string(10)
    variable_name_b = get_random_string(10)


    return f"let {variable_name_a} = {a};\nlet {variable_name_b} = {b};\n"

def generate_socket_operation():
    op_code = random.randint(0, 3)
    function_name = get_random_string(10)
    socket_name = get_random_string(10)
    function_imports = f"use std::net::UdpSocket;"
    fuction_code = f"""
    fn {function_name}() {{
        let mut buffer = [0; 10];
        let {socket_name} = UdpSocket::bind("0.0.0.0:34254").expect("couldn't bind to address");
        {socket_name}.recv_from(&mut buffer).expect("didn't receive data");
    }}\n
    """
    function_call = f"{function_name}();\n"
    return function_imports, fuction_code, function_call


def get_project_root():
    return Path(__file__).parent.parent.parent

def generate_struct():

    struct_name = get_random_string(10)
    struct_code = f"""struct {struct_name} {{"""
    struct_declaration = f"let {get_random_string(10)} = {struct_name} {{"
    fields = random.randint(1, 10)
    for _ in range(fields):
        field_type = random.choice(["i32", "i64", "f32", "f64", "String"])
        field_name = get_random_string(10)
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
    struct_declaration += "\n};"
    struct_code += "\n}\n"
    return struct_code,struct_declaration

def generate_enum():
    enum_name = get_random_string(10)
    enum_code = f"""enum {enum_name} {{"""
    enum_declaration = f"let {get_random_string(10)} = {enum_name}::"
    variants = random.randint(1, 10)
    enum_fields = []
    for _ in range(variants):
        enum_field = get_random_string(10) 
        enum_fields.append(enum_field)

    for field in enum_fields:
        enum_code += f"\n    {field},"
    enum_code += "\n}"
    enum_declaration += f"{random.choice(enum_fields)};"
    return enum_code,enum_declaration

def generate_dataset():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--size", type=int, default=1000)
    argparser.add_argument("--train", type=bool, default=True)
    args = argparser.parse_args()
    size = args.size
    for i in range(size):
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



