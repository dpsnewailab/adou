import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()

setup(
    name="adou",
    version="0.0.0",
    description="Just some typical approaches for document understanding and related tasks.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/dpsnewailab/adou",
    author="DPS-AI Lab",
    author_email="aiteam@dps.com.vn",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=('example', 'test')),
    include_package_data=True,
    install_requires=['torch==1.4.0',
                      'torchvision==0.5.0',
                      'tqdm==4.32.2'],
)