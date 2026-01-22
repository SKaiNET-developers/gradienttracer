# core.py

import os
import importlib.util
from typing import Any, Callable, Dict, List, Tuple, Union
from types import ModuleType

from gt.dot.dag2dot import dag_2_dot
from gt.pytorch.io.writer import store_experiment_as_gguf
from gt.pytorch.trace import trace


def Executable(description: str, op_type: str = None, op_params: Dict[str, Any] = None) -> Callable:
    """Decorator to mark functions as executable with a description.

    Args:
        description: A string describing the purpose of the decorated function.
        op_type: The operation type for validation (e.g., 'conv2d', 'relu', 'flatten').
                 If not provided, the function name will be used.
        op_params: Optional dictionary of operation parameters (e.g., {'padding': 1, 'stride': 2, 'groups': 3}).

    Returns:
        Callable: A decorator function that adds executable attributes.
    """
    def decorator(func: Callable) -> Callable:
        func.executable = True
        func.description = description
        func.op_type = op_type  # None means use function name
        func.op_params = op_params or {}  # Empty dict if not provided
        return func

    return decorator


def load_module_from_file(module_name: str, file_path: str) -> ModuleType:
    """Load a Python module from a file path.

    Args:
        module_name: Name to assign to the loaded module.
        file_path: Path to the Python file to load.

    Returns:
        ModuleType: The loaded Python module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_executable_functions(module: ModuleType) -> List[Tuple[Callable, str, str, Dict[str, Any]]]:
    """Find all functions in a module marked with the @Executable decorator.

    Args:
        module: Python module to search for executable functions.

    Returns:
        List[Tuple[Callable, str, str, Dict]]: List of tuples containing (function, description, op_type, op_params).
    """
    executables = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if callable(attr) and hasattr(attr, 'executable'):
            op_type = getattr(attr, 'op_type', None)
            op_params = getattr(attr, 'op_params', {})
            executables.append((attr, attr.description, op_type, op_params))
    return executables


def iterate_and_execute(folder_path: str) -> List[Dict[str, Any]]:
    """Recursively find and execute all marked functions in test suites.

    Args:
        folder_path: Root directory containing test suite folders.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing execution results and metadata.
    """
    results = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if dir_name.startswith("TS-"):
                ts_number = dir_name.split('-')[1]
                ts_path = str(os.path.join(root, dir_name))
                for file_name in os.listdir(ts_path):
                    if file_name.startswith("UC-") and file_name.endswith(".py"):
                        uc_number = file_name.split('-')[1].split('.')[0]
                        file_path = os.path.join(ts_path, file_name)
                        module_name = f"TS_{ts_number}_UC_{uc_number}"
                        module = load_module_from_file(module_name, file_path)
                        executable_functions = find_executable_functions(module)
                        for func, description, op_type, op_params in executable_functions:
                            inputs, result = func()
                            results.append({
                                'name': f"{module_name}.{func.__name__}",
                                'description': description,
                                'op_type': op_type,  # None if not specified
                                'op_params': op_params,  # Empty dict if not specified
                                'inputs': inputs,
                                'result': result,
                                'test_suite': f"TS-{ts_number}",
                                'use_case': f"UC-{uc_number}"
                            })
    return results


def exec_and_store(folder_path: str, output_path: str, generate_dot:bool == False) -> None:
    """Execute all test cases and store results in GGUF format with visualization.

    Args:
        folder_path: Root directory containing test suite folders.
        output_path: Directory where results will be stored.

    Returns:
        None
    """
    experiment_results = iterate_and_execute(folder_path)
    for result in experiment_results:
        ts_folder = os.path.join(output_path, result['test_suite'])
        os.makedirs(ts_folder, exist_ok=True)
        gguf_file_path = os.path.join(ts_folder, f"{result['name']}.gguf")
        tensors = {f"input_{i}": tensor for i, tensor in enumerate(result['inputs'])}
        # Use op_type if specified, otherwise extract function name (after the last .)
        if result.get('op_type'):
            op_name = result['op_type']
        else:
            op_name = result['name'].split('.')[-1] if '.' in result['name'] else result['name']
        store_experiment_as_gguf(
            experiment_description=result['description'],
            tensors=tensors,
            operation_callback=lambda *args: result['result'],
            gguf_file_path=gguf_file_path,
            operation_name=op_name,
            op_params=result.get('op_params', {})
        )

        if generate_dot:
            graph = trace(result['result'])
            dot = dag_2_dot(graph)
            dot_file_path = os.path.join(ts_folder, f"{result['use_case']}_{result['name']}.dot")
            dot.render(dot_file_path)
