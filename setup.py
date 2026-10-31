"""
[ZH] EPGM 算法库安装配置文件。
[EN] EPGM Algorithm Library setup configuration.
"""

from setuptools import setup, find_packages

setup(
    name="epgm-algorithm",
    version="0.1.0",
    author="EPGM Research Team",
    description="Event-predictive Graph Machine: A non-gradient topological computing mechanism for embodied intelligence",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "gymnasium>=0.29.0",
        "pygame>=2.5.0",
        "matplotlib>=3.7.0",
        "torch>=2.0.0",
    ],
)
