import os
import tempfile
import unittest

import gguf
import torch

from gt.pytorch.io.writer import store_experiment_as_gguf


def _store(path, op_params):
    x = torch.ones(2, 2)
    store_experiment_as_gguf(
        experiment_description="op_params round trip",
        tensors={"input_0": x},
        operation_callback=lambda t: t,
        gguf_file_path=path,
        operation_name="metric.accuracy",
        op_params=op_params,
    )


class TestOpParams(unittest.TestCase):
    def test_string_param_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "case.gguf")
            _store(path, {"average": "macro", "dim": 1, "threshold": 0.5})

            reader = gguf.GGUFReader(path)
            field = reader.get_field("op.average")
            self.assertIsNotNone(field)
            self.assertEqual(gguf.GGUFValueType.STRING, field.types[0])
            self.assertEqual("macro", bytes(field.parts[field.data[0]]).decode("utf-8"))
            self.assertIsNotNone(reader.get_field("op.dim"))
            self.assertIsNotNone(reader.get_field("op.threshold"))

    def test_unsupported_param_type_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "case.gguf")
            with self.assertRaises(TypeError) as raised:
                _store(path, {"labels": {"a": 1}})
            self.assertIn("labels", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
