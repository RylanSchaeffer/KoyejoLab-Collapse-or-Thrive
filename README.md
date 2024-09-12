# KoyejoLab Re-Revisiting Model Collapse

## Setup

(Optional) Update conda:

`conda update -n base -c defaults conda`

Create a conda environment with the required packages:

`conda env create --file environment.yml`

To activate the environment:

`conda activate model_collapse_20240911`

Upgrade pip:

`pip install --upgrade pip`

## Running

This code has two alternating steps: (1) training+evaluation and (2) sampling.

For developing or manually running training+evaluation, from the project directory, run:

```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
python -u src/sft/train_sft.py
```

For developing or manually running sampling, from the project directory, run:

```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
python -u src/sample/sample.py
```

Both load the default hyperparameters from `src/globals.py` and log data to W&B.
The default hyperparameters can be overwritten by W&B sweeps in the directory `sweeps/`.


## Contributing

Please run `black` before committing e.g., `black .`. This ensures we have a consistent formatting style.

## Citations

To cite this work, please use:

## Contact

Questions? Comments? Interested in collaborating? Open an issue or email rschaef@cs.stanford.edu and sanmi@cs.stanford.edu.