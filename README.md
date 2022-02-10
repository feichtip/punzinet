# Punzinet

[![DOI](https://zenodo.org/badge/392275679.svg)](https://zenodo.org/badge/latestdoi/392275679)

A PyTorch implementation of [Punzi-loss: A non-differentiable metric approximation for sensitivity optimisation in the search for new
particles][1].

The given example utilises a simple feedforward neural network which is trained with the Punzi-loss function.
For more information on the Punzi figure of merit and its use see the original paper by Giovanni Punzi: [arXiv:physics/0308063][2].

## Requirements
*    numpy>=1.18.0
*    scipy>=1.4.1
*    torch>=1.5.0
*    pandas>=1.0.0
*    tqdm>=4.17.0 (optional)

## How to use

The given implementation is based on the search for an invisibly decaying Z' in the process <img src="https://render.githubusercontent.com/render/math?math=e^%2be^-\to\mu^%2b\mu^-Z\text{'}">.
Even though the details of the usage can be very analysis dependent, the provided code can be easily adapted for any other new physics search with multiple signal hypotheses, as long as the data is prepared in the right format.
In the present case the mass of the Z' as well as the cross section are free parameters of the search. The signal was generated with a mass starting from 100 MeV up to 10 GeV, in 100 MeV steps, and the main SM backgrounds are <img src="https://render.githubusercontent.com/render/math?math=e^%2be^-\to\mu^%2b\mu^-(\gamma)">, <img src="https://render.githubusercontent.com/render/math?math=e^%2be^-\to\tau^%2b\tau^-"> and <img src="https://render.githubusercontent.com/render/math?math=e^%2be^-\to e^%2be^-\mu^%2b\mu^-">.

### Prepare training data

The training data should be provided in a pandas DataFrame with the following columns:
* `category`:  the background category (e.g. 'mumu') or in case of signal the generated mass
* `range_idx_low`: index of the first bin in which the event appears
* `range_idx_high`: index of the last bin in which the event appears
* `gen_mass`: the generated mass for signals (in case of background set to -999)
* `sig_m_range`: 1 if the signal hypotheses is used for the training (in the mass range), 0 otherwise
* `labels`: 0 for background, 1 for signal
* `M`: the reconstructed mass of the Z' (recoil mass)
* `weights`: a weight factor to account for different sample sizes and luminosity


| category   |   range_idx_low |   range_idx_high |   gen_mass |   sig_m_range |   labels |       M |   weights |
|------------|-----------------|------------------|------------|---------------|----------|---------|-----------|
| mumu       |               6 |                7 |       -999 |             0 |        0 | 1.3465  | 0.111111  |
| 1800MeV    |               9 |                9 |       1800 |             0 |        1 | 1.89452 | 0.550186  |
| mumu       |              10 |               10 |       -999 |             0 |        0 | 2.10327 | 0.111111  |
| 2400MeV    |              11 |               12 |       2400 |             0 |        1 | 2.41797 | 0.550186  |
| 1700MeV    |              18 |               18 |       1700 |             1 |        1 | 3.64961 | 0.550186  |
| taupair    |              26 |               26 |       -999 |             0 |        0 | 5.25229 | 0.0166667 |
| taupair    |              26 |               26 |       -999 |             0 |        0 | 5.29312 | 0.0166667 |
| 5900MeV    |              29 |               29 |       5900 |             1 |        1 | 5.9116  | 0.550186  |
| eemumu     |              42 |               42 |       -999 |             0 |        0 | 8.49311 | 0.05      |
| eemumu     |              43 |               43 |       -999 |             0 |        0 | 8.70446 | 0.05      |


### Example dataset

Example data that follows the mentioned scheme and was used to derive the results in the paper are available at `examples/training_data.feather` (hosted by Git Large File Storage).


### Pretrain model

In a first step the model has to be pretrained using a conventional loss function such as BCE. This is implemented in `train.bce_training()`.

### Run the training with the Punzi-loss function

Now we can use the pretrained network and continue training with the Punzi-loss function. This is implemented in `train.punzi_training()`.
The loss is calculated simultaneously for all mass hypotheses by utilising sparse matrices in `fom.py`.

A full working example with training data is provided in the examples directory.
In this case the training with the Punzi-loss is performed without dividing the sample into batches.
When the training is not consistent and depends a lot on the inital choice of hyperparameters, additional batching can help to escape local minima.

[1]: https://doi.org/10.1140/epjc/s10052-022-10070-0
[2]: https://arxiv.org/abs/physics/0308063 "Sensitivity of searches for new signals and its optimization"
