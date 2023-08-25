# AutoML lecture 2023 (Freiburg & Hannover)
## Final Project
Choice: Meta-Learning - Using prior data to warmstart optimization

This is our team's approach at finding a combination of search algorithm and multi-fidelity scheduler that performs well on the dataset `deepweeds`.

The search algorithm is a Bayesian Optimizer that employs a random forest regressor as a surrogate model.
This surrogate model is warmstarted on objective function evaluations that were supplied.

The scheduler employs a prediction and grace period strategy to create unique adaptive fidelity behavior.

## Installation

Download the Repository

```bash
git clone https://github.com/MarcSpeckmann/Meta-Learning-Using-Prior-Data-to-Warmstart-Optimization.git
```

First you need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html#system-requirements).

Make sure there is no environment that is already named "automl"

Subsequently, run these commands:
```bash
make install
```

Activate the environment
```bash
conda activate automl
```

You may want to adjust the settings for how many concurrent trials to run and how many CPUs and GPUs to use per trial. These settings are found starting at line 281 in `main.py`

Our test system was able to handle up to about 12 concurrent trials, with fractional resource allocations.

Run the main experiment file
```bash
python main.py
```

### Experiments

For reproducing the experiments detailed on the poster run the specific `experiment_<searcher>_<scheduler>_<1,2,3>.py`

### Results

The test set accuracy will automatically be displayed after finishing the set runtime.
