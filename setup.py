import os
from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [
            line.strip() for line in f
            if line and not line.startswith('#') and '--index-url' not in line
        ]

def install_special_dependencies():
    os.system('pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118')
    os.system('pip install torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118')

setup(
    name="open_magvit2",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description='Open-source replication of Google\'s MAGVIT-v2 tokenizer',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  
    url="https://github.com/TencentARC/Open-MAGVIT2",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'configs': ['configs/gpu/*.yaml', 'configs/npu/*.yaml'],
    },
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)