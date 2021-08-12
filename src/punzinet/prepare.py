import pandas as pd
import numpy as np


def set_weights(df, scale):
    df['weights'] = 1.0
    df.loc[~df.signal, 'weights'] = df[~df.signal].category.map(scale)
    signal_factor = df[~df.signal].weights.sum() / df[df.signal].weights.sum()
    df.loc[df.signal, 'weights'] *= signal_factor
    df['weights'] = df.weights.astype('single')


def set_range_index(df, m_range, width, n_sigma=2):
    m_gev = ((m_range / 1000).reshape(-1, 1))
    w = width.loc[m_range].w.values

    low_bound = (m_gev**2 - n_sigma * w.reshape(-1, 1))
    high_bound = (m_gev**2 + n_sigma * w.reshape(-1, 1))
    in_range = (low_bound < df.M2.values) & (df.M2.values < high_bound)

    range_idx_low = np.ones(len(df), dtype=np.int) * in_range.shape[0] + 100
    range_idx_high = np.ones(len(df), dtype=np.int) * -2

    for i, row in enumerate(in_range):
        range_idx_low[row & (range_idx_low > i)] = i
        range_idx_high[row & (range_idx_high < i)] = i

    range_idx_low[range_idx_low == in_range.shape[0] + 100] = -1

    df['sig_m_range'] = (df.gen_mass.map({mass: 1 for mass in m_range}) == 1).astype('int')
    df['range_idx_low'] = range_idx_low
    df['range_idx_high'] = range_idx_high


def bin_widths(file):
    widths = pd.read_csv(file, sep="\t")
    widths.columns = ['m', 'w', 'e']
    widths = widths.set_index('m')

    return widths
