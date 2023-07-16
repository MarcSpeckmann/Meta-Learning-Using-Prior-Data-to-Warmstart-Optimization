[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/T_Fzxg5j)
# AutoML lecture 2023 (Freiburg & Hannover)
## Final Project

This repository contains all things needed for the final projects.
The task is to optimize a NN by AutoML means.
For details, please refer to the project PDF.

###  Install

First you need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html#system-requirements).

Subsequently, run these commands:
```bash
make install
```

### Data
You need to pre-download all the data required by running `make download-data`.

Stores by default in a `./data` directory. Takes under 20 seconds to download and extract.

### Tips

All code we provide does consider validation and training sets.
You will have to implement a method to use the test set yourself.

#### `meta_learning_template.py`
* Example of how to run SMAC.
* Provides hints for how to extend SMAC to warmstart with meta-learning.
* Provides code to read and parse the meta-data.

#### Plotting
* We do not provide plotting scripts for the examples.
  You are allowed to use everything you already know from the lecture.
  We recommend to use 'matplotlib' or 'seaborn' for plotting.
* To get an example of how to plot SMAC data (which we used in the example code), you can take a look at
the [SMAC3 documentation](https://automl.github.io/SMAC3).
  An example similar to our multi-fidelity example can be found [here](https://automl.github.io/SMAC3/v2.0.1/examples/2_multi_fidelity/1_mlp_epochs.html).
* You are free to implement and use any other plotting scripts.
