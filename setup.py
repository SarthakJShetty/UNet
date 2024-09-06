from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

parsed_requirements = [requirement in requirements if "--index-url" not in requirement]

setup(
    name="unet",
    version="0.1",
    description="a simple unet implementation",
    author="sarthak",
    author_email="sarthakshetty97@gmail.com",
    packages=find_packages(),
    install_requires=parsed_requirements,
)
