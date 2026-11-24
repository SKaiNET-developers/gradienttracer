import os
import tempfile
import unittest

import gguf
import torch

from gt.pytorch.io.writer import store_experiment_as_gguf


def read_tensor(reader: gguf.GGUFReader, tensor_index: int) -> torch.Tensor:
    for index, tensor in enumerate(reader.tensors):
        if index == tensor_index:
            shape = tensor.shape
            reshaped_array = tensor.data.reshape(shape)

            return torch.from_numpy(reshaped_array)


def read_description(reader):
    for field in reader.fields:
        if 'description' in field:
            field = reader.get_field(field)
            byte_data = field.parts[field.data[0]].tobytes()
            # Decode the bytes object to a string
            decoded_string = byte_data.decode('utf-8')
            return decoded_string
    return ""


class TestExperiment(unittest.TestCase):
    def test_experiment(self):
        x = torch.randn(3, 3)  # First operand
        y = torch.randn(3, 3)  # Second operand

        # Define a callback for addition
        def addition_op(tensor1, tensor2):
            return tensor1 + tensor2

        with tempfile.TemporaryDirectory() as tmp:
            gguf_file_path = os.path.join(tmp, 'test_experiment.gguf')

            # Tensors are passed by name; the callback gets them in dict order
            store_experiment_as_gguf(
                "Simple addition", {"input_0": x, "input_1": y}, addition_op, gguf_file_path
            )

            # Read back the data from the gguf file
            reader = gguf.GGUFReader(gguf_file_path)
            self.assertEqual(
                ["input_0", "input_1", "result"], [t.name for t in reader.tensors]
            )
            read_operand1 = read_tensor(reader, 0)
            read_operand2 = read_tensor(reader, 1)
            description = read_description(reader)
            read_result = read_tensor(reader, 2)

        # Assert the read values are equal to the original values
        self.assertTrue(torch.allclose(x, read_operand1))
        self.assertTrue(torch.allclose(y, read_operand2))
        self.assertEqual('Simple addition', description)
        self.assertTrue(torch.allclose(addition_op(x, y), read_result))
