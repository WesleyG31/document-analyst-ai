from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements=f.read().splitlines()
    
    setup(
        name="document_analysis",
        version="0.0.1",
        author="Wesley Gonzales",
        packages=find_packages(),
        install_requires=requirements
    )

# pip install -e .