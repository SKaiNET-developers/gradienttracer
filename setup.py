import os

from setuptools import setup, find_packages

project_name = "gt"


def get_version():
    """Read version from ``__init__.py``."""
    version_filepath = os.path.join(os.path.dirname(__file__), project_name, "__init__.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


setup(
    name="gradienttracer",
    author="Michal Harakal",
    version=get_version(),
    packages=find_packages(),
    install_requires=['click==8.1.7', 'gguf==0.18.0', 'numpy==2.0.0', 'sentencepiece==0.2.0',
                      'torch==2.3.1', 'tqdm==4.66.4', 'typer==0.12.3', 'typing_extensions==4.12.2',
                      'typer-cli==0.12.3', 'graphviz==0.20.3'],
    python_requires=">=3.8",
    package_data={'': []},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'gt=gt.exec:app'
        ]
    }
)
