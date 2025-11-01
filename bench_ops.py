import jinja2
import os
import sys
import subprocess
import pandas as pd


def list_available_unary_operations():

    import operations

    all_operations = operations.UNARY_OPERATIONS
    all_operations = [(variant_name, base_operation_name) for variant_name, base_operation_name, _, _ in operations.iterate_all_operations(all_operations)]


    return all_operations


def generate_all_operations_templates(all_operations, dest_dir, template_file, ITERATIONS, DTYPE):
    """
    For each operation in all_operations

    all_operations: dict of {
        "base_operation_name": str, 
        "implementation_name": str
    }
    """

    os.makedirs(dest_dir, exist_ok=True)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("templates"),
        autoescape=True
    )

    template = env.get_template(template_file)

    all_impl_files = []

    for implementation_name, base_operation_name in all_operations:
        rendered_template = template.render(ITERATIONS=ITERATIONS, DTYPE=DTYPE, IMPLEMENTATION_NAME=implementation_name)

        impl_file = f"{dest_dir}/{base_operation_name}-{DTYPE}-{implementation_name}.py"
        with open(impl_file, "w") as f:
            f.write(rendered_template)
            all_impl_files += [(impl_file, base_operation_name, implementation_name)]

    return all_impl_files

def run_bench(implementation, dest_dir):

    impl_file, base_operation_name, implementation_name = implementation

    # Launch tracy for the given operation file
    subprocess.run(["tracy", impl_file], check=True)

    # Parse results

    # Pick best result
    # That is, row where 'DEVICE KERNEL DURATION PER CORE MIN [ns]' is the smallest
    # Then, compute the following metrics:
    # 1. Cycles per datum
    # 2. Cycles per tile

    # Return data frame with 
    # [base_operation_name, implementation_name, cycles_per_datum, cycles_per_tile]

    pass

def run_benchmarks(all_implementations, dest_dir):
    
    df_all_results = pd.DataFrame()
    for impl_file, base_operation_name, implementation_name in all_implementations:
        run_bench(impl_file, base_operation_name, implementation_name, dest_dir)

    pass



def main(args):

    all_operations = list_available_unary_operations()

    os.makedirs("generated/benchmarks/unary/", exist_ok=True)

    generate_all_operations_templates(all_operations, "generated/benchmarks/unary/", "bench-unary.py.j2", 10, "bfloat16")

    pass

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)