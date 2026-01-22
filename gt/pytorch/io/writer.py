import numpy
import torch
import gguf
import inspect
import numpy as np


def __ensure_little_endian__(tensor: torch.Tensor) -> torch.Tensor:
    """Ensures the tensor is in Little Endian format."""
    if tensor.dtype.byteorder not in ('<', '=', '|'):
        tensor = tensor.byteswap().newbyteorder('<')
    return tensor


def __convert_to_f32__(tensor: torch.Tensor) -> numpy.ndarray:
    """Converts the tensor to float32 if it is not already."""
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    return tensor


def store_experiment_as_gguf(experiment_description: str, tensors: dict, operation_callback, gguf_file_path: str, operation_name: str = None, op_params: dict = None):
    """
    Perform a mathematical operation on an array of tensors and store the operands, operator, and result in a gguf file.

    :param experiment_description: stores a description of the experiment
    :param tensors: Dictionary containing tensor names and their values
    :param operation_callback: Callback function to perform the operation
    :param gguf_file_path: Path to the gguf file to store the results
    :param operation_name: Name of the operation (if None, uses operation_callback.__name__)
    :param op_params: Dictionary of operation parameters (e.g., {'padding': 1, 'stride': 2})
    """
    # Convert tensors to Little Endian format after detaching them
    tensors_le = {name: __convert_to_f32__(__ensure_little_endian__(tensor.detach().numpy())) for name, tensor in tensors.items()}

    # Perform the operation using the callback and detach the result tensor
    result = operation_callback(*[tensor.detach() for tensor in tensors.values()])
    result_le = __convert_to_f32__(__ensure_little_endian__(result.detach().numpy()))

    # Use provided operation_name, fallback to callback's __name__
    name = operation_name if operation_name else operation_callback.__name__

    # Prepare data to write into gguf file
    writer = gguf.GGUFWriter(gguf_file_path, arch='llama')
    writer.add_description(experiment_description)
    for name_key, tensor_le in tensors_le.items():
        writer.add_tensor(name_key, tensor_le)
    writer.add_name(name)

    # Store operation parameters as custom metadata
    if op_params:
        for param_name, param_value in op_params.items():
            key = f"op.{param_name}"
            if isinstance(param_value, int):
                if param_value < 0:
                    writer.add_int32(key, param_value)
                else:
                    writer.add_uint32(key, param_value)
            elif isinstance(param_value, float):
                writer.add_float32(key, param_value)
            elif isinstance(param_value, (list, tuple)):
                # Store as array of ints (for stride, padding tuples)
                writer.add_array(key, [int(v) for v in param_value])

    writer.add_tensor("result", result_le)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Experiment data stored in {gguf_file_path}")