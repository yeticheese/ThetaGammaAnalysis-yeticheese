from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ThetaGammaAnalysis',
    version='1.0.0',
    description='Theta Gamma Analysis Dependency Packages Installer',
    packages=find_packages(),
    install_requires=requirements,
)