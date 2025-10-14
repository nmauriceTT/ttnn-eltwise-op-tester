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

def iterate_all_operations(operation_dict):
    for category, operations in operation_dict.items():

        if "golden" in operations:
            golden_operation = operations["golden"]
        else:
            for impl_name, implementation in operations["implementations"].items():
                try:
                    golden_function = ttnn.get_golden_function(impl_name)
                except:
                    continue
                
                golden_operation = implementation.golden_function
                break

        if golden_operation is None:
            print(f"{TERM_RED}No golden operation found for category {category}{TERM_RESET}")
            continue

        for impl_name, implementation in operations["implementations"].items():
            yield (impl_name, implementation, golden_operation)

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



UNARY_OPERATIONS_LEGACY = {
    # Exponential functions
    "abs": (torch.abs, ttnn.abs, None, "abs"),
    "identity": (lambda x, out: torch.nn.Identity()(x), ttnn.identity, None, "identity"),
    "fill": (
        lambda x, out: torch.fill(out, 1.99999988079071044921875),
        lambda x, output_tensor: ttnn.fill(x, 1.99999988079071044921875, output_tensor=output_tensor), 
        None, 
        "fill"
    ),
    "exp": (torch.exp, ttnn.exp, math.exp, "exp"),
    "exp-approx": (
        torch.exp,
        lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "exp",
    ),
    "exp-fast-approx": (
        torch.exp,
        lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "exp",
    ),
    "exp-fast-approx-v2": (
        torch.exp,
        lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "exp",
    ),
    "exp_cond": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp_approx0": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp21f": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp_21f_round_nearest": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp_hybrid": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "tanh": (
        torch.tanh,
        ttnn.tanh,
        math.tanh,
        "tanh",
    ),  # ttnn.tanh() does not support output_tensor ?
    "tanh-approx": (
        torch.tanh,
        lambda x, output_tensor: ttnn.tanh(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        math.tanh,
        "tanh",
    ),
    "tanh_accurate": (
        torch.tanh,
        lambda x, output_tensor: ttnn.tanh_accurate(x, accurate_mode=True, output_tensor=output_tensor),
        math.tanh,
        "tanh",
    ),
    "cosh": (
        torch.cosh,
        lambda x, output_tensor: ttnn.cosh(x),
        math.cosh,
        "cosh",
    ),  # ttnn.cosh() does not support output_tensor ?
    "sinh": (
        torch.sinh,
        lambda x, output_tensor: ttnn.sinh(x),
        math.sinh,
        "sinh",
    ),  # ttnn.sinh() does not support output_tensor ?
    # Logarithmic functions
    "log": (torch.log, ttnn.log, math.log, "log"),
    "log10": (torch.log10, ttnn.log10, math.log10, "log10"),
    "log2": (torch.log2, ttnn.log2, math.log2, "log2"),
    "log1p": (torch.log1p, ttnn.log1p, math.log1p, "log1p"),
    "logaddexp": (torch.logaddexp, ttnn.logaddexp, None, "logaddexp"),
    "logaddexp2": (torch.logaddexp2, ttnn.logaddexp2, None, "logaddexp2"),
    # Activation functions
    "silu": (lambda x, out: torch.nn.SiLU()(x), ttnn.silu, None, "silu"),
    "gelu": (lambda x, out: torch.nn.GELU()(x), ttnn.gelu, None, "gelu"),
    "gelu_approx": (
        lambda x, out: torch.nn.GELU()(x),
        lambda x, output_tensor: ttnn.gelu(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "gelu",
    ),
    "logit": (
        torch.logit,
        lambda x, output_tensor: ttnn.logit(x),
        None,
        "logit",
    ),  # ttnn.logit does not support output_tensor ?
    "swish": (
        lambda x, out: torch.nn.SiLU()(x),
        lambda x, output_tensor: ttnn.swish(x),
        None,
        "swish",
    ),  # ttnn.swish does not support output_tensor ?
    "mish": (lambda x, out: torch.nn.Mish()(x), ttnn.mish, None, "mish"),
    "elu": (
        lambda x, out: torch.nn.ELU()(x),
        lambda x, output_tensor: ttnn.elu(x, output_tensor=output_tensor, alpha=1.0),
        None,
        "elu",
    ),  # Unlike torch, ttnn.elu does not use alpha=1 by default
    "celu": (
        lambda x, out: torch.nn.CELU()(x),
        lambda x, output_tensor: ttnn.celu(x, output_tensor=output_tensor, alpha=1.0),
        None,
        "celu",
    ),
    "selu": (
        lambda x, out: torch.nn.SELU()(x),
        lambda x, output_tensor: ttnn.selu(x),
        None,
        "selu",
    ),  # ttnn.selu does not support output_tensor ?
    "softplus": (lambda x, out: torch.nn.Softplus()(x), ttnn.softplus, None, "softplus"),
    "softsign": (
        lambda x, out: torch.nn.Softsign()(x),
        lambda x, output_tensor: ttnn.softsign(x),
        None,
        "softsign",
    ),  # ttnn.softsign does not support output_tensor ?
    # Trigonometric functions
    "tan": (torch.tan, ttnn.tan, math.tan, "tan"),
    "atan": (torch.atan, ttnn.atan, math.atan, "atan"),
    "sin": (torch.sin, ttnn.sin, math.sin, "sin"),
    "cos": (torch.cos, ttnn.cos, math.cos, "cos"),
    # Miscellaneous functions
    "sqrt": (torch.sqrt, ttnn.sqrt, math.sqrt, "sqrt"),
    "cbrt": (
        lambda x, out: torch.pow(x, 1/3), 
        lambda x, output_tensor: ttnn.cbrt(x), 
        None, 
        "cbrt"),
    "cbrt-pow1d3": (
        lambda x, out: torch.pow(x, 1/3), 
        lambda x, output_tensor: cbrt_pow1d3(x, output_tensor), 
        None, 
        "cbrt"),
    "cbrt-pow1d3-fp32": (
        lambda x, out: torch.pow(x, 1/3), 
        lambda x, output_tensor: run_ttnn_fp32_and_round_bf16(ttnn.cbrt, [x]), 
        None, 
        "cbrt"),
    "rsqrt": (
        torch.rsqrt,
        lambda x, output_tensor: ttnn.rsqrt(x, output_tensor=output_tensor),
        None,
        "rsqrt",
    ),
    "rsqrt_approx": (
        torch.rsqrt,
        lambda x, output_tensor: ttnn.rsqrt(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "rsqrt",
    ),
    "reciprocal": (
        torch.reciprocal,
        ttnn.reciprocal,
        None,
        "reciprocal",
    ),
    "digamma": (
        torch.digamma,
        lambda x, output_tensor: ttnn.digamma(x),
        None,
        "digamma",
    ),  # ttnn.digamma does not support output_tensor ?
    "lgamma": (
        torch.lgamma,
        lambda x, output_tensor: ttnn.lgamma(x),
        math.lgamma,
        "lgamma",
    ),  # ttnn.lgamma does not support output_tensor ?
    "tanhshrink": (
        lambda x, out: torch.nn.Tanhshrink()(x),
        lambda x, output_tensor: ttnn.tanhshrink(x),
        None,
        "tanhshrink",
    ),  # ttnn.tan
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
    "pow": {
        "ttnn": ttnn.pow,
        "torch": torch.pow,
    },
    "pow21f": {
        "ttnn": ttnn.pow,
        "torch": torch.pow,
    },
    "divide": {
        "ttnn": ttnn.divide,
        "torch": torch.div,
    },
    "divide-sfpu": {
        "ttnn": lambda x, y: run_ttnn_fp32_and_round_bf16(ttnn.divide, [x, y]),
        "torch": torch.div,
    },
    "div": {
        "ttnn": ttnn.div,
        "torch": torch.div,
    },
    "div-accurate": {
        "ttnn": lambda x, y: ttnn.div(x, y, accurate_mode=True),
        "torch": torch.div,
    },
}




def run_ttnn_op(fun, args, device):
    global global_device
    global_device = device

    ttnn_args = [convert_to_ttn(arg, device) for arg in args if isinstance(arg, torch.Tensor)]

    return fun(*ttnn_args)