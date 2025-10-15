import ttnn
import torch
import math
from utils import TERM_RED, TERM_RESET


global_device = None

def run_op_fp32(ttnn_op, args, device=None):
    ttnn_args = [convert_to_ttn(arg, device) for arg in args]
    return ttnn_op(*ttnn_args)


def cbrt_pow1d3(x, out):

    ttnn_1d3 = ttnn.full_like(x, 1/3)
    return ttnn.pow(x, ttnn_1d3)


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
            "exp_cond": ttnn.exp,
            "exp_approx0": ttnn.exp,
            "exp21f": ttnn.exp,
            "exp_21f_round_nearest": ttnn.exp,
            "exp_hybrid": ttnn.exp
        },
    },
    "tanh": {
        "implementations": {
            "tanh": ttnn.tanh,
            "tanh-approx": lambda x, output_tensor: ttnn.tanh(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
            "tanh_accurate": lambda x, output_tensor: ttnn.tanh_accurate(x, accurate_mode=True, output_tensor=output_tensor)
        },
    },
    "cosh": {
        "implementations": {
            "cosh": lambda x, output_tensor: ttnn.cosh(x)
        },
    },
    "sinh": {
        "implementations": {
            "sinh": lambda x, output_tensor: ttnn.sinh(x)
        },
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
            "log2": ttnn.log2
        },
    },
    "log1p": {
        "implementations": {
            "log1p": ttnn.log1p
        },
    },
    "logaddexp": {
        "implementations": {
            "logaddexp": ttnn.logaddexp
        },
    },
    "logaddexp2": {
        "implementations": {
            "logaddexp2": ttnn.logaddexp2
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
            "gelu_approx": lambda x, output_tensor: ttnn.gelu(x, fast_and_approximate_mode=True, output_tensor=output_tensor)
        },
    },
    "logit": {
        "implementations": {
            "logit": lambda x, output_tensor: ttnn.logit(x)
        },
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
    },
    "celu": {
        "implementations": {
            "celu": lambda x, output_tensor: ttnn.celu(x, output_tensor=output_tensor, alpha=1.0)
        },
    },
    "selu": {
        "implementations": {
            "selu": lambda x, output_tensor: ttnn.selu(x)
        },
    },
    "softplus": {
        "implementations": {
            "softplus": ttnn.softplus
        },
    },
    "softsign": {
        "implementations": {
            "softsign": lambda x, output_tensor: ttnn.softsign(x)
        },
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
            "cbrt-pow1d3": lambda x, output_tensor: cbrt_pow1d3(x, output_tensor),
            "cbrt-pow1d3-fp32": lambda x, output_tensor: run_ttnn_fp32_and_round_bf16(ttnn.cbrt, [x])
        },
    },
    "rsqrt": {
        "implementations": {
            "rsqrt": lambda x, output_tensor: ttnn.rsqrt(x, output_tensor=output_tensor),
            "rsqrt_approx": lambda x, output_tensor: ttnn.rsqrt(x, fast_and_approximate_mode=True, output_tensor=output_tensor)
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
    },
    "lgamma": {
        "implementations": {
            "lgamma": lambda x, output_tensor: ttnn.lgamma(x)
        },
    },
    "tanhshrink": {
        "implementations": {
            "tanhshrink": lambda x, output_tensor: ttnn.tanhshrink(x)
        },
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
    "pow21f": {
        "implementations": {
            "pow21f": ttnn.pow
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
    }
}


def iterate_all_operations(operation_dict):
    """ Iterate over all operations in the operation dictionary 
    and yield the implementation name, the implementation function, and the golden function
    """
    for category, operations in operation_dict.items():

        golden_function = None
        if "golden" in operations:
            golden_function = operations["golden"]
        else:
            for impl_name, implementation in operations["implementations"].items():
                try:
                    golden_function = ttnn.get_golden_function(impl_name)
                except:
                    print(f"{TERM_RED}No golden function found for implementation {impl_name}{TERM_RESET}")
                    continue
                
                golden_function = implementation.golden_function
                break

        if golden_function is None:
            print(f"{TERM_RED}No golden operation found for category {category}{TERM_RESET}")
            continue

        for impl_name, implementation in operations["implementations"].items():
            yield (impl_name, implementation, golden_function)

def get_operation_by_name(operation_dict, impl_name):

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

def get_operations_by_category(operation_dict, category):

    if not category in operation_dict:
        raise ValueError(f"Category {category} not found in operation_dict")

    return operation_dict[category]







def run_ttnn_op(fun, args, device):
    global global_device
    global_device = device

    ttnn_args = [convert_to_ttn(arg, device) for arg in args if isinstance(arg, torch.Tensor)]

    return fun(*ttnn_args)