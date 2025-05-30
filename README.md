# MAMGD_Optimizer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?logo=Keras&logoColor=white)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Gradient optimization method using exponential damping and second-order discrete derivative for neural networks and multidimensional real functions

* Sakovich, N.; Aksenov, D.; Pleshakova, E.; Gataullin, S. MAMGD: Gradient-Based Optimization Method Using Exponential Decay. Technologies 2024, 12, 154. https://doi.org/10.3390/technologies12090154

## Gradient-based Optimization Method for Multidimensional Real Functions and Neural Networks

This project focuses on the development of a gradient-based optimization method for multidimensional real functions and neural networks using exponential decay with Tensorflow and Keras.

### Abstract
Optimization methods, namely, gradient optimization methods, are a key part of neural network training. In this paper, we propose a new gradient optimization method using exponential decay and the adaptive learning rate using a discrete second-order derivative of gradients. The MAMGD optimizer uses an adaptive learning step, exponential smoothing and gradient accumulation, parameter correction, and some discrete analogies from classical mechanics. The experiments included minimization of multivariate real functions, function approximation using multilayer neural networks, and training neural networks on popular classification and regression datasets. The experimental results of the new optimization technology showed a high convergence speed, stability to fluctuations, and an accumulation of gradient accumulators. The research methodology is based on the quantitative performance analysis of the algorithm by conducting computational experiments on various optimization problems and comparing it with existing methods.

## Technologies Used
- Tensorflow
- Keras
- Matplotlib
- NumPy
- Pandas
- Optuna

## Features
- Optimization of multidimensional real functions
- Optimization of neural networks
- Utilizes exponential decay for optimization
- Integration with Tensorflow and Keras for seamless implementation

## Algorithm
<img width="800px" src="https://github.com/NekkittAY/MAMGD_Optimizer/blob/main/doc/MAMGD_optimizer_img.jpg?raw=true"/>

## Article 
```
@article{sakovich2024mamgd,
  title={MAMGD: Gradient-Based Optimization Method Using Exponential Decay},
  author={Sakovich, Nikita and Aksenov, Dmitry and Pleshakova, Ekaterina and Gataullin, Sergey},
  journal={Technologies},
  volume={12},
  number={9},
  pages={154},
  year={2024},
  publisher={MDPI}
}
```
