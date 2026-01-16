import numpy as np
import torch
import traceback

import pytest
from models.common.utility_functions import ulp


TERM_RED = "\033[91m"
TERM_GREEN = "\033[92m"
TERM_RESET = "\033[0m"


def all_bf16_tensor():
    size = [2**9, 2**7]

    input_np = np.arange(0, 2**16, dtype=np.uint16)
    print(f"input_np: ({input_np.size}) \n{input_np}")

    torch_values = torch.from_numpy(input_np.view(np.int16)).reshape(size)
    torch_values_bf16 = torch_values.view(torch.bfloat16)

    return torch_values_bf16


def generate_all_f32_tensors():

    # We want to test all possible float32 values. 
    # Since storing everything in a single tensor is not possible, we return a generator 
    # that yields tensors of size [2**8, 2**9, 2**7] 
    # For float32, that amounts to 64 MB per tensor, which is still reasonable

    width = 2**7
    height = 2**9
    batch_size = 2**6

    num_elements = batch_size * height * width

    num_tensors = 2**32 // num_elements
    shape = [batch_size, height, width]

    tensor = torch.arange(0, num_elements, dtype=torch.int64).reshape(shape)
    increment = num_elements

    print(f"num_batch: {num_tensors}")

    for _ in range(num_tensors):
        tensor_f32 = tensor.to(torch.int32).view(torch.float32)

        yield tensor_f32
        tensor += increment


def generate_binary_tensors_bf16():
    batch_size = 128
    num_batches = 2**16 // batch_size

    shape = [batch_size, 2**9, 2**7]

    all_bf16 = all_bf16_tensor()
    tensor_a = all_bf16.reshape([1, 2**9, 2**7]).expand(batch_size, -1, -1)

    # each slice of tensor_b contains the same value
    # tensor_b cotnains 128 slices. Each slice has a unique value.
    # Value(slice[i]) = nextafter(Vallue(slice[i-1]))

    tensor_b_i16 = torch.arange(-(2**16), -(2**16) + batch_size, dtype=torch.int16)
    assert tensor_b_i16.shape == torch.Size([batch_size])

    tensor_b_i16 = tensor_b_i16.reshape([batch_size, 1, 1]).repeat(1, 2**9, 2**7)
    assert tensor_b_i16.shape == tensor_a.shape

    increment = batch_size

    for i in range(0, 2**16, batch_size):
        tensor_b = tensor_b_i16.view(torch.bfloat16)

        yield (tensor_a, tensor_b)

        tensor_b_i16 += increment


def execute_benchmarks(benchmark_fun, operations, dest_dir):
    success_count = 0
    successfull_operations = []
    failed_operations = []

    cnt = 0
    total_operation_cnt = 0 # Set to 0 for now because operations is a generator (can't get len without iterating on it)
    for operation in operations:
        cnt += 1
        print(f"Running operation {operation}  #{cnt} / {total_operation_cnt}", end="\r")
        try:
            benchmark_fun(operation, dest_dir)
            success_count += 1
            successfull_operations += [operation]
        except Exception as e:
            print(f"{TERM_RED}Could not run operation {operation}: {e}{TERM_RESET}")
            print(f"{TERM_RED}{traceback.format_exc()}{TERM_RESET}")
            failed_operations += [operation]

    return (successfull_operations, failed_operations)
