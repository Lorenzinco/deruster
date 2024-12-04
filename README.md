<p align="center">
    <img src="logo.png" width="200">
</p>

# DeRuster an AI-powered rust decompiler

This project came up when i was following the advanced machine learning 2024 course at Sapienza university of Rome.

## Compy
DeRuster is proudly powered by [Compy](https://github.com/Etto48/compy), please refer to the documentation to apply changes.

## The dataset folder
The project's scripts need a folder named "data", inside this folder divide the datasets into two other folders: test and training.
The training and validation sets are split automatically from the test folder inside two subsets, which the validation is TRAIN_SIZE/100*VALIDATION_SPLIT, validation split is set at 20%, it can be modified inside the dataset.py file.


## Running the model
This project makes use of python scripts, to run the model on a given binary please type in terminal:
```sh
run <binary_path>
```
To retrain the model type:
```sh
train
```

## Dataset
A dataset generator is provided, to generate functions and variables inside the data folder please type in terminal:
```sh
generate --size <n-programs>
```

To compile such dataset into the asm folder please type in terminal:
```sh
compile
```