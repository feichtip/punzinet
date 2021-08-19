# Punzinet

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
In the present case the mass of the Z' as well as the cross section are free parameters of the search. The signal was generated with as mass starting from 100 MeV up to 10 GeV, in 100 MeV steps and the main SM backgrounds are <img src="https://render.githubusercontent.com/render/math?math=e^%2be^-\to\mu^%2b\mu^-(\gamma)">, <img src="https://render.githubusercontent.com/render/math?math=e^%2be^-\to\tau^%2b\tau^-"> and <img src="https://render.githubusercontent.com/render/math?math=e^%2be^-\to e^%2be^-\mu^%2b\mu^-">.

### Prepare training data

The training data should be provided in a pandas DataFrame with the following columns present:
* category
* range_idx_low
* range_idx_high
* gen_mass
* sig_m_range
* labels
* M
* weights


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


### Pretrain model

### Run the training with the Punzi-loss function


[1]: link_to_paper
[2]: https://arxiv.org/abs/physics/0308063 "Sensitivity of searches for new signals and its optimization"
