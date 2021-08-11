import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib


def catlist(df, var, cats, scale=1):
    return [df.loc[df.category == cat, var] * scale for cat in cats]


def output_distribution(df, net, X, cut_overlay=None, mass_range=None, ylim=None, bkgs=[], norm='log'):
    y_score = net(torch.from_numpy(X)).detach().numpy()[:, 0]

    if ylim is None:
        ylim = (df.M.min(), df.M.max())
    xlim = (y_score.min(), y_score.max())

    if len(bkgs) == 0:
        selections = [~df.signal.values, df.signal.values]
        labels = ['background events', 'signal events']
    else:
        selections = [df.category == bkg for bkg in bkgs]
        selections.append(df.signal.values)
        labels = bkgs + ['signal']

    for selection, label in zip(selections, labels):
        bins, x_edges, y_edges = np.histogram2d(y_score[selection], df.M.values[selection], bins=70, range=[xlim, ylim])
        p = plt.pcolor(x_edges, y_edges, bins.T, cmap='viridis', shading='flat', norm=matplotlib.colors.LogNorm())
        p.set_edgecolor("face")
        plt.colorbar()

        if cut_overlay is not None and mass_range is not None:
            plt.plot(cut_overlay, mass_range / 1000, color='red', marker='')

        plt.title(label, fontdict={'fontsize': 18})
        plt.xlabel('classification variable')
        plt.ylabel(r'M$_{\sf{rec}}$ [GeV/c$^2$]')
        plt.tight_layout()
        plt.show()
