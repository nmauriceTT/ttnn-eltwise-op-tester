import ttnn
import torch
from .utils import TERM_RED, TERM_RESET

from .kernel_generator import generate_unary_kernel_from_polynomial, generate_unary_kernel_from_sfpi_source, generic_unary_kernel, generate_kernel_source_code_from_llk


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
            # "exp-approx": lambda x, output_tensor: generic_unary_kernel(generate_kernel_source_code_from_llk("unary", "exp_tile_init<true, false>", "exp_tile<true, false>"), x, output_tensor),
            "exp-fast-approx": lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
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
            "expm1": ttnn.expm1,
        },
    },
    "tanh": {
        "implementations": {
            "tanh": ttnn.tanh,
        },
        "golden": torch.tanh
    },
    "hardtanh": {
        "implementations": {
            "hardtanh": lambda x, output_tensor: ttnn.hardtanh(x)
        },
        "golden": lambda x, out: torch.nn.functional.hardtanh(x)
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
            "log": ttnn.log,
        },
        "golden": torch.log,
    },
    "log10": {
        "implementations": {
            "log10": ttnn.log10
        },
    },
    "log2": {
        "implementations": {
            "log2": ttnn.log2,
        },
    },
    "log1p": {
        "implementations": {
            "log1p": ttnn.log1p
        },
    },
    "relu": {
        "implementations": {
            "relu": ttnn.relu
        },
    },
    "relu_max": {
        "implementations": {
            "relu_max": lambda x, output_tensor: ttnn.relu_max(x, output_tensor=output_tensor, upper_limit=1.0)
        },
        "golden": lambda x, out: torch.minimum(torch.nn.functional.relu(x), torch.full_like(x, 1.0)) 
    },
    "relu_min": {
        "implementations": {
            "relu_min": lambda x, output_tensor: ttnn.relu_min(x, output_tensor=output_tensor, lower_limit=1.0)
        },
        "golden": lambda x, out: torch.maximum(torch.nn.functional.relu(x), torch.full_like(x, 1.0)) 
    },
    "silu": {
        "implementations": {
            "silu": ttnn.silu,
        },
    },
    "gelu": {
        "implementations": {
            "gelu": ttnn.gelu,
            "gelu_approx": lambda x, output_tensor: ttnn.gelu(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
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
    "hardmish": {
        "implementations": {
            "hardmish": ttnn.hardmish,
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
    "acos": {
        "implementations": {
            "acos": ttnn.acos
        },
        "golden": torch.acos
    },
    "asin": {
        "implementations": {
            "asin": ttnn.asin
        },
        "golden": torch.asin
    },
    "atan": {
        "implementations": {
            "atan": ttnn.atan,
        },
        "golden": torch.atan
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
    },
    "erfinv": {
        "implementations": {
            "erfinv": lambda x, output_tensor: ttnn.erfinv(x)
        },
        "golden": lambda x, out: torch.special.erfinv(x)
    },
    "round": {
        "implementations": {
            "round": lambda x, output_tensor: ttnn.round(x, output_tensor=output_tensor)
        },
        "golden": torch.round
    },
    "ceil": {
        "implementations": {
            "ceil": lambda x, output_tensor: ttnn.ceil(x, output_tensor=output_tensor)
        },
        "golden": torch.ceil
    },
    "floor": {
        "implementations": {
            "floor": lambda x, output_tensor: ttnn.floor(x, output_tensor=output_tensor)
        },
        "golden": torch.floor
    },
}

BINARY_OPERATIONS = {
    "add": {
        "implementations": {
            "add": ttnn.add
        },
        "golden": torch.add
    },
    "subtract": {
        "implementations": {
            "subtract": ttnn.subtract
        },
        "golden": torch.subtract
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
            "div-accurate": lambda x, y: ttnn.div(x, y, accurate_mode=True)
        },
        "golden": torch.div
    },
    "fmod": {
        "implementations": {
            "fmod": ttnn.fmod
        },
        "golden": torch.fmod
    },
    "atan2": {
        "implementations": {
            "atan2": ttnn.atan2,
        },
        "golden": torch.atan2
    },
    "rsub": {
        "implementations": {
            "rsub": ttnn.rsub
        },
        "golden": torch.rsub
    },
    "subtract": {
        "implementations": {
            "subtract": ttnn.subtract
        },
        "golden": torch.subtract
    },

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
