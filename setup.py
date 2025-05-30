from setuptools import setup, find_packages

setup(
    name="revisiting_cropa",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "tqdm",
        "matplotlib",
        "pandas",
        "jupyter",
    ],
    python_requires=">=3.8",
    author="Project Authors",
    author_email="your.email@example.com",
    description="Revisiting CroPA: A comprehensive study of cross-prompt attacks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/Revisiting_CroPA",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 