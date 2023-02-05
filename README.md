# Structural Credit Assignment in Neural Networks Using Reinforcement Learning

## Introduction

The credit assignment problem is that a network should assign credit or blame for its behaviours according to the contribution to the network. In neural networks, structural credit assignment (SCA) refers to the problem of determining which neurons or specific parts of the network are responsible for a certain output or behaviour of its performance. A widely used solution for SCA is Backpropagation. However, Reinforcement Learning also can be used to address this problem. 
The implementation of this project is based on [this](https://proceedings.neurips.cc/paper/2021/file/fe1f9c70bdf347497e1a01b6c486bdb9-Paper.pdf) paper. 
## Directory Structure

![Folder-Structure](/folder-structure.png)

# How to use

- Clone the Repository into your project directory
- Open the terminal and cd into project directory
```
cd <path to the project directory>
```
- Create a virtual environment
```
python -m venv <path to the environment>
```
- Activate the environment
```
<path to the environment>\Scripts\activate 
```
- cd into the project directory
```
cd <path to the project directory>
```
- Install all the requirements using following commad
```
pip install -r requirements.txt
```
- Use the following command to execute the code
```
python train.py -conf config.json -out_path ./runs
```
- runs and data directories will be created automatically.
- Check the runs directory to find plots, pt files and csv files.
- Use config.json to change the settings
- Use following commad to evaluate the model
```
python eval.py <path to the model file (pt file)>
```

