from setuptools import setup

setup(
    name="unet",
    version="0.1",
    description="simple unet architecture",
    author="sarthak",
    author_email="sarthakshetty97@gmail.com",
    packages=["unet"],
    install_requires=[
        "matplotlib==3.9.0",
        "tqdm",
        "matplotlib-inline==0.1.7",
    ],
)
