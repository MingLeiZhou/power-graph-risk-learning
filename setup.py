from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="power-graph-risk-learning",
    version="0.1.0",
    author="MingLeiZhou",
    description=(
        "A data-driven framework for risk assessment and digital twin "
        "modeling in power systems using graph-based learning"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MingLeiZhou/power-graph-risk-learning",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "networkx>=2.6",
        "scikit-learn>=1.0",
        "pandas>=1.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
