<br/>
<p align="center">
  <h3 align="center">Theta Gamma Analysis</h3>

  <p align="center">
    A Theta Gamma Analysis Repository hosted at the
<a href="https://www.genzellab.com/">Genzel Lab</a>
    <br/>
    <br/>
  </p>
</p>

![Contributors](https://img.shields.io/github/contributors/AbdelRayan/ThetaGammaAnalysis?color=dark-green) 

## Table Of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)

[//]: # (* [Roadmap]&#40;#roadmap&#41;)

[//]: # (* [Contributing]&#40;#contributing&#41;)

[//]: # (* [License]&#40;#license&#41;)
[//]: # (* [Authors]&#40;#authors&#41;)
* [References](#references)

## About The Project
[//]: # (![Screen Shot]&#40;images/screenshot.png&#41;)
This repository is a culmination of all the techniques and methods used to discover the properties of the Theta-Gamma 
code. Currently, we're working on a combination of techniques used in two papers (Lopes-dos-Santos et al., 2018 & Zhang et al., 2019).
An ongoing project at the Donders Institute for Brain Cognition and Behaviour, Radboud Univeristy, this project will periodically go through updates
as the analysis is being refined, stay tuned for more developments.


## Getting Started

Most of the analysis is carried out within Python environments, tutorial notebooks can be loaded independently using
Google Colab or locally,except for Tutorial 4. The following steps will guide you through the installation process

## Prerequisites
### Install Python:
 * For Windows,  it's recommended to download and install Python directly from the official Python website.

Make sure to check the "Add Python to PATH" option during installation so that you can run Python from the command prompt.
 * For macOS (using Homebrew):
   ```bash
   brew install python
   ```
### Verify Python Installation:
```bash
   python --version
````
### Installation

### Clone The Repository
  * Download the repository from the github repo web address

  * **OR** Clone it using the terminal.

    Navigate to the directory where you want to clone the repository:
```bash
cd /path/to/your/desired/directory
```
```bash
git clone https://github.com/AbdelRayan/ThetaGammaAnalysis.git
cd ThetaGammaAnalysis
```
### Install the dependencies
 * Through the `requirements.txt` file
```bash
pip install -r requirements.txt
```
 * **OR** Through the `setup.py` file
```bash
python setup.py install
```
## Usage

Please go through Tutorials 1 through 4 on how the code is implemented and patched together. 
This section will be further detailed with more usage scenarios and implementations

# References
* [1. Lopes-dos-Santos, V., van de Ven, G. M., Morley, A., Trouche, S., Campo-Urriza, N., &amp; Dupret, D. (2018). Parsing hippocampal theta oscillations by nested spectral components during spatial exploration and memory-guided behavior. Neuron, 100(4). doi:10.1016/j.neuron.2018.09.031](https://www.cell.com/neuron/fulltext/S0896-6273(18)30833-X) 
* [2. Zhang, L., Lee, J., Rozell, C., &amp; Singer, A. C. (2019). Sub-second dynamics of theta-gamma coupling in hippocampal CA1. eLife, 8. doi:10.7554/elife.44320 ](ttps://elifesciences.org/articles/44320)