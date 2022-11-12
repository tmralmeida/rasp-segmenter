from setuptools import setup, find_namespace_packages

setup(
    name="rasp-segmenter",
    version="0.1.0",
    description="Segmentation of raspberries",
    license="MIT",
    author="Tiago Almeida",
    author_email="tmr.almeida96@gmail.com",
    python_requires=">=3.9.0",
    url="https://github.com/tmralmeida/rasp-segmenter",
    packages=find_namespace_packages(
        exclude=["tests", ".tests", "tests_*", "scripts"]),
    classifiers=[
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)