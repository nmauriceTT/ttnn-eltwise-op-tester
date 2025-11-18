import ttnn
import torch
import math
from utils import TERM_RED, TERM_RESET

from kernel_generator import generate_unary_kernel_from_polynomial, generate_unary_kernel_from_sfpi_source, generic_unary_kernel


global_device = None

def run_op_fp32(ttnn_op, args, device=None):
    ttnn_args = [convert_to_ttn(arg, device) for arg in args]
    return ttnn_op(*ttnn_args)


def convert_to_ttn(tensor, device):

    if isinstance(tensor, ttnn.Tensor):
        return tensor

    # Shard data on all cores to speed-up computation
    ttnn_tensor = ttnn.from_torch(tensor, layout=ttnn.Layout.TILE, device=device)
    
    return ttnn_tensor

def run_ttnn_fp32_and_round_bf16(ttnn_op, args):

    global global_device
    device = args[0].device
    assert device is not None

    host_bf16_args = [convert_to_ttn(arg, device) for arg in args]
    host_f32_args = [arg.to(torch.float32) for arg in host_bf16_args]
    ttnn_args = [ttnn.from_torch(arg, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) for arg in host_f32_args]

    result = ttnn_op(*ttnn_args)
    
    # Convert back to bf16
    result = ttnn.to_torch(result)
    result = result.to(torch.bfloat16)
    result = ttnn.from_torch(result, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return result

# No golden function set => use ttnn.get_golden_function() from first implementation
UNARY_OPERATIONS = {
    "abs": {
        "implementations": {
            "abs": ttnn.abs
        },
        "golden": torch.abs
    },
    "identity": {
        "implementations": {
            "identity": ttnn.identity 
        },
    },
    "fill": {
        "implementations": {
            "fill": lambda x, output_tensor: ttnn.fill(x, 1.99999988079071044921875, output_tensor=output_tensor)
        },
        "golden": lambda x, out: torch.fill(out, 1.99999988079071044921875)
    },
    "exp": {
        "implementations": {
            "exp": ttnn.exp,
            "exp-approx": lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "exp-fast-approx": lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "exp-fast-approx-v2": lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "exp-21f": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("exp", [0.33718944, 0.65763629, 1.0017248], "exp-21f"), x, output_tensor),
            "exp-61f": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("exp", [0.0002170391, 0.001243946, 0.0096788315, 0.055483369, 0.24022982, 0.69314699, 1.0000000018], "exp-61f"), x, output_tensor),
            "exp-Chebyshev-v1[2]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("exp", [0.34228965640068054,0.652752697467804,1.0022648572921753], "exp-Chebyshev-v1[2]"), x, output_tensor),
            "exp-Chebyshev-v1[4]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("exp", [0.013670309446752071,0.05174499750137329,0.24160435795783997,0.6929728984832764,1.000003457069397], "exp-Chebyshev-v1[4]"), x, output_tensor),
            "exp-Chebyshev-v1[6]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("exp", [0.00021865784947294742,0.0012391331838443875,0.009684186428785324,0.055480629205703735,0.24023045599460602,0.6931469440460205,1.0], "exp-Chebyshev-v1[6]"), x, output_tensor),
            "exp-Chebyshev-v1-c0ef0[4]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("exp", [0.012763113714754581,0.05344102904200554,0.24064704775810242,0.6931340098381042,1.0], "exp-Chebyshev-v1-c0ef0[4]"), x, output_tensor),

        },
        "golden": torch.exp,
    },
    "exp2": {
        "implementations": {
            "exp2": ttnn.exp2
        },
    },
    "expm1": {
        "implementations": {
            "expm1": ttnn.expm1
        },
    },
    "tanh": {
        "implementations": {
            "tanh": ttnn.tanh,
            "tanh-approx": lambda x, output_tensor: ttnn.tanh(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "tanh-cf": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("tanh-v1"), x, output_tensor),
            "tanh-pade-5,5":  lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("tanh-pade-5,5"), x, output_tensor),
            "tanh-minimax-v1[6]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("tanh-minimax-v1[6]"), x, output_tensor),
            # Other approximations:
            # "tanh-Chebyshev-v1-c0ef0[6]": [0.004613510798662901,-0.0569886788725853,0.25763407349586487,-0.46735504269599915,0.02672632411122322,0.9987236261367798,0.0],
            # "tanh-minimax-v0[4]": [2.49048434197902679443359375e-2, -8.3681561052799224853515625e-2, -0.20078647136688232421875,1.0220668315887451171875, 0.0],
            # "tanh-minimax-v1[5]": [-1.950809545814990997314453125e-2, 0.1467897593975067138671875, -0.325587689876556396484375, -4.27231900393962860107421875e-2, 1.00523841381072998046875, 0.0],
        },
        "golden": torch.tanh
    },
    "cosh": {
        "implementations": {
            "cosh": lambda x, output_tensor: ttnn.cosh(x)
        },
        "golden": torch.cosh
    },
    "sinh": {
        "implementations": {
            "sinh": lambda x, output_tensor: ttnn.sinh(x)
        },
        "golden": torch.sinh
    },
    "log": {
        "implementations": {
            "log": ttnn.log
        },
    },
    "log10": {
        "implementations": {
            "log10": ttnn.log10
        },
    },
    "log2": {
        "implementations": {
            "log2": ttnn.log2,
            "log2-minimax-v1[3]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("log2-poly", [0.2044459879398345947265625, -0.6402385234832763671875, 1.43861830234527587890625, 0.0], "log2-minimax-v1[3]"), x, output_tensor),
            "log2-minimax-v1[5]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("log2-poly", [5.96523843705654144287109375e-2, -0.22712136805057525634765625, 0.441900312900543212890625, -0.7169878482818603515625, 1.44261324405670166015625, 0.0], "log2-minimax-v1[5]"), x, output_tensor),
        },
    },
    "log1p": {
        "implementations": {
            "log1p": ttnn.log1p
        },
    },
    "silu": {
        "implementations": {
            "silu": ttnn.silu,
            "swish": lambda x, output_tensor: ttnn.swish(x)  # swish is same as silu
        },
    },
    "gelu": {
        "implementations": {
            "gelu": ttnn.gelu,
            "gelu_approx": lambda x, output_tensor: ttnn.gelu(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "gelu-tanh": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("gelu-tanh"), x, output_tensor),
        },
    },
    "logit": {
        "implementations": {
            "logit": lambda x, output_tensor: ttnn.logit(x)
        },
        "golden": torch.logit
    },
    "mish": {
        "implementations": {
            "mish": ttnn.mish
        },
    },
    "elu": {
        "implementations": {
            "elu": lambda x, output_tensor: ttnn.elu(x, output_tensor=output_tensor, alpha=1.0)
        },
        "golden": lambda x, out: torch.nn.functional.elu(x)
    },
    "celu": {
        "implementations": {
            "celu": lambda x, output_tensor: ttnn.celu(x, output_tensor=output_tensor, alpha=1.0)
        },
        "golden": lambda x, out: torch.nn.functional.celu(x)
    },
    "sigmoid": {
        "implementations": {
            "sigmoid": ttnn.sigmoid,
            "sigmoid-approx": lambda x, output_tensor: ttnn.sigmoid(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "sigmoid-accurate": ttnn.sigmoid_accurate,
            "sigmoid-accurate-approx": lambda x, output_tensor: ttnn.sigmoid_accurate(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "sigmoid-21f": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("sigmoid"), x, output_tensor),
        },
    },
    "log_sigmoid": {
        "implementations": {
            "log_sigmoid": ttnn.log_sigmoid
        },
    },
    "selu": {
        "implementations": {
            "selu": lambda x, output_tensor: ttnn.selu(x)
        },
        "golden": lambda x, out: torch.nn.functional.selu(x)
    },
    "softplus": {
        "implementations": {
            "softplus": ttnn.softplus,
            "softplus-log1pexp": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_sfpi_source("softplus-log1pexp"), x, output_tensor),
            "softplus-minimax-v1[5]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("softplus-poly", [-1.88397825695574283599853515625e-4, -1.717669540084898471832275390625e-3, 5.2007441408932209014892578125e-3, 0.113285191357135772705078125, 0.476007401943206787109375, 0.689675033092498779296875]), x, output_tensor),
            "softplus-minimax-v1[8]": lambda x, output_tensor: generic_unary_kernel(generate_unary_kernel_from_polynomial("softplus-poly", [7.496111464888599584810435771942138671875e-8, 7.1853546614875085651874542236328125e-6, 5.31854093424044549465179443359375e-5, -2.879406674765050411224365234375e-4, -3.30807245336472988128662109375e-3, 3.11028095893561840057373046875e-3, 0.121423818171024322509765625, 0.4927570819854736328125, 0.692435443401336669921875]), x, output_tensor),
        },
    },
    "softsign": {
        "implementations": {
            "softsign": lambda x, output_tensor: ttnn.softsign(x)
        },
        "golden": lambda x, out: torch.nn.functional.softsign(x)
    },
    "tan": {
        "implementations": {
            "tan": ttnn.tan
        },
    },
    "atan": {
        "implementations": {
            "atan": ttnn.atan
        },
    },
    "sin": {
        "implementations": {
            "sin": ttnn.sin
        },
    },
    "cos": {
        "implementations": {
            "cos": ttnn.cos
        },
    },
    "sqrt": {
        "implementations": {
            "sqrt": ttnn.sqrt
        },
    },
    "cbrt": {
        "implementations": {
            "cbrt": lambda x, output_tensor: ttnn.cbrt(x),
        },
        "golden": lambda x, out: torch.pow(x, 1/3)
    },
    "rsqrt": {
        "implementations": {
            "rsqrt": ttnn.rsqrt
        },
    },
    "reciprocal": {
        "implementations": {
            "reciprocal": ttnn.reciprocal
        },
    },
    "digamma": {
        "implementations": {
            "digamma": lambda x, output_tensor: ttnn.digamma(x)
        },
        "golden": torch.digamma
    },
    "lgamma": {
        "implementations": {
            "lgamma": lambda x, output_tensor: ttnn.lgamma(x)
        },
        "golden": torch.lgamma
    },
    "tanhshrink": {
        "implementations": {
            "tanhshrink": lambda x, output_tensor: ttnn.tanhshrink(x)
        },
        "golden": lambda x, out: torch.nn.functional.tanhshrink(x)
    }
}


def divide_sfpu(x, y):
    assert isinstance(x, ttnn.Tensor)
    assert isinstance(y, ttnn.Tensor)

    # Convert on host to ensure proper rounding
    # device = x.device
    layout = x.layout

    global global_device
    device = global_device
    assert device is not None

    host_bf16_x = ttnn.to_torch(x)
    host_bf16_y = ttnn.to_torch(y)

    # Convert to float32
    host_f32_x = host_bf16_x.to(torch.float32)
    host_f32_y = host_bf16_y.to(torch.float32)

    ttnn_x = ttnn.from_torch(host_f32_x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_y = ttnn.from_torch(host_f32_y, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.divide(ttnn_x, ttnn_y)

    # Convert back to bf16
    result = ttnn.to_torch(result)
    result = result.to(torch.bfloat16)
    result = ttnn.from_torch(result, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return result


BINARY_OPERATIONS = {
    "add": {
        "implementations": {
            "add": ttnn.add
        },
        "golden": torch.add
    },
    "multiply": {
        "implementations": {
            "multiply": ttnn.multiply
        },
        "golden": torch.multiply
    },
    "hypot": {
        "implementations": {
            "hypot": ttnn.hypot
        },
        "golden": torch.hypot
    },
    "pow": {
        "implementations": {
            "pow": ttnn.pow
        },
        "golden": torch.pow
    },
    "divide": {
        "implementations": {
            "divide": ttnn.divide,
            "div": ttnn.div,
            "divide-sfpu": lambda x, y: run_ttnn_fp32_and_round_bf16(ttnn.divide, [x, y]),
            "div-accurate": lambda x, y: ttnn.div(x, y, accurate_mode=True)
        },
        "golden": torch.div
    },
    "atan2": {
        "implementations": {
            "atan2": ttnn.atan2
        },
        "golden": torch.atan2
    },
    "rsub": {
        "implementations": {
            "rsub": ttnn.rsub
        },
        "golden": torch.rsub
    }
}


def get_op_implementations(operation_dict, category):

    if not category in operation_dict:
        raise ValueError(f"Category {category} not found in operation_dict")

    return operation_dict[category]

def get_golden_function(operation_dict, operation_name: str):
    golden_function = None
    operation = get_op_implementations(operation_dict, operation_name)
    if "golden" in operation:
        golden_function = operation["golden"]
    else:
        for impl_name, implementation in operation["implementations"].items():
            try:
                golden_function = ttnn.get_golden_function(implementation)
            except:
                print(f"{TERM_RED}No golden function found for implementation {impl_name}{TERM_RESET}")
                continue
            
            golden_function = implementation.golden_function
            break

    if golden_function is None:
        raise ValueError(f"No golden operation found for operation {operation_name}")

    return golden_function
    

def iterate_all_operations(operation_dict):
    """ Iterate over all operations in the operation dictionary 
    and yield the implementation name, base operation name the implementation function, and the golden function
    """
    for base_operation_name, operations in operation_dict.items():

        golden_function = get_golden_function(operation_dict, base_operation_name)
        for impl_name, implementation in operations["implementations"].items():
            yield (impl_name, base_operation_name, implementation, golden_function)

def get_operation_variant_by_name(operation_dict, impl_name):

    for category, operations in operation_dict.items():
        if impl_name in operations["implementations"].keys():

            impl = operations["implementations"][impl_name]

            if "golden" in operations:
                golden_operation = operations["golden"]
            else:
                ttnn_impl = getattr(ttnn, category)
                golden_operation = ttnn.get_golden_function(ttnn_impl)

            return impl, golden_operation

    return None



def run_ttnn_op(fun, args, device):
    global global_device
    global_device = device

    ttnn_args = [convert_to_ttn(arg, device) for arg in args if isinstance(arg, torch.Tensor)]

    return fun(*ttnn_args)