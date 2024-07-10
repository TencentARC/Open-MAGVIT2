from setuptools import find_namespace_packages, setup

def parse_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

setup(
    name="magvit2",
    version="1.0",
    packages=find_namespace_packages(include=['taming.*']),
    install_requires=parse_requirements('requirements.txt'),
)
