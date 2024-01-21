# Lab 3.2: BadNet
## Introduction
this project is a reproduction code for [BadNet](https://arxiv.org/pdf/1708.06733.pdf), 
aimed at reproducing experiments on the MNIST dataset in the paper
## Usage
### Dataset
please download the [MINIT](https://github.com/geektutu/tensorflow-tutorial-samples/tree/master/mnist/data_set) dataset in directory `./data/MNIST/raw`
### Example
if you'd like to test single target attack, firstly generate the poisoned dataset.
you need to set model.CCNN.config.mode equals 'ij', and set ij_class what you want, like [0, 1], then run this:
```
python src/load_dataset.py
```
the poisoned data file will be saved in directory `./data/PoisonedMNIST/ij/`

after that, run:
```
python src/main.py --action 1
```
The result will be output in the console and saved in directory `./log/ij/`:

If you need to test other experiments, please check the `action` parameter in `./src/main.py`