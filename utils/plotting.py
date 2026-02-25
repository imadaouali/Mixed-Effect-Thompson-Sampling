"""Plotting utilities: linestyle conversion and regret plots."""

import numpy as np


def linestyle2dashes(style):
    """Convert matplotlib linestyle string to dash tuple."""
    if style == "--":
        return (3, 3)
    if style == ":":
        return (0.5, 2.5)
    return (None, None)


def plot_regret(results, algs_config, title, save_path=None):
    """
    Plot cumulative regret (mean + error bars at 10 points).
    results: list of (regret 2d array) per algorithm, same order as algs_config.
    algs_config: list of (name, params, color, linestyle, label).
    """
    import matplotlib.pyplot as plt
    from utils.plotting import linestyle2dashes

    for (regret, alg_cfg) in zip(results, algs_config):
        _, _, color, linestyle, label = alg_cfg
        cum = regret.cumsum(axis=0)
        steps = cum.shape[0]
        step = np.arange(1, steps + 1)
        sube = (steps // 10) * np.arange(1, 11) - 1
        sube = sube[sube < steps]
        plt.plot(
            step,
            cum.mean(axis=1),
            color,
            dashes=linestyle2dashes(linestyle),
            label=label,
        )
        plt.errorbar(
            step[sube],
            cum[sube, :].mean(axis=1),
            cum[sube, :].std(axis=1) / np.sqrt(cum.shape[1]),
            fmt="none",
            ecolor=color,
        )
    plt.title(title)
    plt.xlabel("Round n")
    plt.ylabel("Regret")
    plt.ylim(bottom=0)
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
