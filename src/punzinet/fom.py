import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import tqdm


def check_type(input):
    if isinstance(input, pd.core.series.Series):
        return torch.from_numpy(input.values)
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    else:
        return input


def punzi_sensitivity(sig_eff, bkg_eve, a=3, b=1.28155, lumi=50, numpy=False):
    sqrt = np.sqrt if numpy else torch.sqrt
    sig_eff[sig_eff == 0] = 1E-6
    return (a**2 / 8 + 9 * b**2 / 13 + a * sqrt(bkg_eve) + b * sqrt(b**2 + 4 * a * sqrt(bkg_eve) + 4 * bkg_eve) / 2) / sig_eff / lumi


def punziloss(sig_sparse, bkg_sparse, outputs, weights, n_gen_signal, target_lumi, scaling=1):
    weights = check_type(weights)
    sig_eff = ((sig_sparse).float().to(outputs.device) @ outputs.reshape(-1, 1) * scaling / n_gen_signal).flatten()
    bkg_eve = ((bkg_sparse).float().to(outputs.device) @ (outputs * weights).reshape(-1, 1) * scaling).flatten()

    fom_array = punzi_sensitivity(sig_eff, bkg_eve, lumi=target_lumi)

    return fom_array


def gen_sparse_matrices(gen_mass, range_idx_low, range_idx_high, sig_m_range, m_range_len, numpy=False):
    """
    memory efficient
    """
    gen_mass = check_type(gen_mass)
    range_idx_low = check_type(range_idx_low)
    range_idx_high = check_type(range_idx_high)
    sig_m_range = check_type(sig_m_range)

    bkg = gen_mass == -999
    sig_m_range = sig_m_range.type(torch.BoolTensor)

    idx_range = (1 + range_idx_high - range_idx_low).byte()
    v1, v2, v1_bkg, v2_bkg = ([], [], [], [])

    for j in range(idx_range.max()):
        indices = torch.nonzero(idx_range > j)[:, 0]
        v1.append(range_idx_low[indices] + j)
        v2.append(indices)

        indices_bkg = torch.nonzero((idx_range > j) & bkg)[:, 0]
        v1_bkg.append(range_idx_low[indices_bkg] + j)
        v2_bkg.append(indices_bkg)

    i = torch.zeros(2, idx_range.sum(), dtype=torch.long)
    i[0, :] = torch.cat(v1)
    i[1, :] = torch.cat(v2)
    v = torch.ByteTensor([1]).expand(i.shape[1])

    _, inverse_indices = torch.unique(gen_mass[sig_m_range], sorted=True, return_inverse=True)
    i_sig = torch.zeros(2, len(inverse_indices), dtype=torch.long)
    i_sig[0, :] = inverse_indices
    i_sig[1, :] = torch.arange(len(gen_mass))[sig_m_range]
    v_sig = torch.ByteTensor([1]).expand(i_sig.shape[1])

    i_bkg = torch.zeros(2, (idx_range * bkg).sum(), dtype=torch.long)
    i_bkg[0, :] = torch.cat(v1_bkg)
    i_bkg[1, :] = torch.cat(v2_bkg)
    v_bkg = torch.ByteTensor([1]).expand(i_bkg.shape[1])

    if numpy:
        sparse_shape = (m_range_len, len(gen_mass))
        in_range = csr_matrix((v.numpy(), i.numpy()), shape=sparse_shape)
        sig_sparse = csr_matrix((v_sig.numpy(), i_sig.numpy()), shape=sparse_shape).multiply(in_range)
        bkg_sparse = csr_matrix((v_bkg.numpy(), i_bkg.numpy()), shape=sparse_shape)
    else:
        sparse_shape = torch.Size([m_range_len, len(gen_mass)])
        in_range = torch.sparse.ByteTensor(i, v, sparse_shape)
        sig_sparse = torch.sparse.ByteTensor(i_sig, v_sig, sparse_shape) * in_range
        bkg_sparse = torch.sparse.ByteTensor(i_bkg, v_bkg, sparse_shape)

    return sig_sparse, bkg_sparse


def optimal_cut(df, output, width, mass_range, n_gen_signal, steps=200, scaling=1, lower_bound=0, upper_bound=1, split_every=20):
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    output = output.reshape(-1, 1)
    sig_sparse, bkg_sparse = gen_sparse_matrices(df.gen_mass, df.range_idx_low, df.range_idx_high, df.sig_m_range, len(mass_range), numpy=True)

    output_weights = df.weights.values.reshape(-1, 1)

    if lower_bound is None:
        lower_bound = output.min()
    if upper_bound is None:
        upper_bound = output.max()

    cut_range = np.linspace(lower_bound, upper_bound, steps)
    sig_eff = np.zeros((len(mass_range), len(cut_range)))
    bkg_eve = np.zeros((len(mass_range), len(cut_range)))
    cut_range_split = torch.split(torch.from_numpy(cut_range), split_every)

    for i, cut in tqdm.tqdm(enumerate(cut_range_split), desc=f'iteration through cut values in steps of {split_every}', total=len(cut_range_split), leave=True, unit='block', position=0):

        non_zero_i = torch.nonzero(output > cut).t()
        idx_start = (i * split_every)
        idx_stop = (i * split_every) + len(cut)

        v_out = torch.ByteTensor([1]).expand(non_zero_i.shape[1]).numpy()
        outputs = csr_matrix((v_out, non_zero_i), shape=(len(output), len(cut))).astype('float')

        sig_eff[:, idx_start:idx_stop] = np.asarray(((sig_sparse @ outputs) / n_gen_signal * scaling).todense())
        bkg_eve[:, idx_start:idx_stop] = np.asarray(((bkg_sparse @ outputs.multiply(output_weights)) * scaling).todense())

    fom_array = punzi_sensitivity(sig_eff, bkg_eve, numpy=True)

    # get the cut indices and FOM values for the lowest Punzi sensitivity
    indices = fom_array.argmin(axis=1)
    values = fom_array.min(axis=1)

    return cut_range[indices], values


def fixed_cut(df, output, width, mass_range, n_gen_signal, cut=0.5, scaling=1):
    if type(output) == np.ndarray:
        output = torch.from_numpy(output)
    output = output.reshape(-1, 1)
    sig_sparse, bkg_sparse = gen_sparse_matrices(df.gen_mass, df.range_idx_low, df.range_idx_high, df.sig_m_range, len(mass_range))

    output_weights = torch.from_numpy(df.weights.values).type(torch.DoubleTensor).reshape(-1, 1)
    outputs = (output > cut).type(torch.DoubleTensor)

    sig_eff = ((sig_sparse.double() @ outputs) / n_gen_signal * scaling)
    bkg_eve = ((bkg_sparse.double() @ (outputs * output_weights)) * scaling)

    fom_array = punzi_sensitivity(sig_eff, bkg_eve)

    return fom_array.flatten().numpy()
