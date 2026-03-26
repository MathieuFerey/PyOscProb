import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

from pyoscprob.oscillation import Oscillator
from pyoscprob.oscillation_parameters import OscillationParameters
from pyoscprob.earth import EarthModel


flavour_to_latex = {
    "e": r"\nu_e",
    "mu": r"\nu_\mu",
    "tau": r"\nu_\tau"
}

flavour_to_latex_anti = {
    "e": r"\bar{\nu}_e",
    "mu": r"\bar{\nu}_\mu",
    "tau": r"\bar{\nu}_\tau"
}


def compute_oscillograms(alpha, Thetaz, E, earth, osc_params, anti=False):
    """
    Compute oscillation probabilities for a given initial flavor alpha
    through all zenith angles Thetaz and energies E.

    Parameters
    ----------
    alpha : str or int
        Initial flavor ("e", "mu", "tau") or corresponding index (0,1,2).
    Thetaz : array_like
        Zenith angles (radians).
    E : astropy.Quantity
        Array of energies with units (e.g., GeV).
    earth : EarthModel
        Earth model for density slabs.
    osc_params : OscillationParameters
        Oscillation parameters.
    anti : bool, optional
        Whether to compute probabilities for the antineutrino flavor (default: False).

    Returns
    -------
    P_alpha : dict
        Dictionary with keys "e", "mu", "tau", each an array of shape
        (len(Thetaz), len(E)), giving the probabilities for that final flavor.
    """

    # Precompute slabs for all theta values
    print("Precomputing slabs for all thetas...")
    Slabs_list = [earth.compute_slabs(theta) for theta in Thetaz]

    # Initialize oscillator
    osc = Oscillator(osc_params)

    # Allocate probability arrays
    N_theta = len(Thetaz)
    N_E = len(E)
    P_alpha = {
        "e": np.zeros((N_theta, N_E)),
        "mu": np.zeros((N_theta, N_E)),
        "tau": np.zeros((N_theta, N_E)),
    }

    # Loop over zenith angles
    print("Computing oscillation probabilities...")
    for i in tqdm(range(N_theta), total=N_theta):

        # Compute probabilities for this theta
        probs = osc.probabilities(alpha, Slabs_list[i], E, anti=anti)

        # Fill arrays
        for flavor in ["e", "mu", "tau"]:
            P_alpha[flavor][i] = probs[flavor]

    print("Done.")

    return P_alpha


# def plot_oscillograms(Thetaz, E, P_alpha, init_flavour, final_flavours=["e", "mu", "tau"], anti=False, title=None):

#     if len(final_flavours) == 0:
#         raise ValueError("At least one final flavor must be specified.")
#     elif len(final_flavours) > 3:
#         raise ValueError("At most three final flavors can be specified.")
#     elif len(final_flavours) == 1:
#         figsize = (6, 5)
#     elif len(final_flavours) == 2:
#         figsize = (12, 5)
#     elif len(final_flavours) == 3:
#         figsize = (18, 5)

#     plt.figure(figsize=figsize)

#     for i, final_flavour in enumerate(final_flavours):

#         plt.subplot(1, len(final_flavours), i+1)

#         plt.pcolormesh(E.value, np.cos(Thetaz), P_alpha[final_flavour], shading="auto", cmap="hot")
#         plt.colorbar(label="Probability")

#         plt.xscale("log")
#         plt.xlabel(r"$E_\nu$" + f" [{E.unit}]")
#         plt.ylabel(r"cos($\theta_z$)")

#         if anti:
#             plt.title(f"${flavour_to_latex_anti[init_flavour]} \\to {flavour_to_latex_anti[final_flavour]}$")
#         else:
#             plt.title(f"${flavour_to_latex[init_flavour]} \\to {flavour_to_latex[final_flavour]}$")

#     if title is not None:
#         plt.suptitle(title)

#     plt.tight_layout()
#     plt.show()

def plot_oscillograms(Thetaz, E, P_alpha, init_flavour,
                     final_flavours=["e", "mu", "tau"],
                     anti=False, title=None,
                     filename="oscillogram.png"):

    if len(final_flavours) == 0:
        raise ValueError("At least one final flavor must be specified.")
    elif len(final_flavours) > 3:
        raise ValueError("At most three final flavors can be specified.")

    # --- Force 16:9 aspect ratio ---
    n = len(final_flavours)
    base_width = 16
    base_height = 9
    figsize = (base_width, base_height)

    fig = plt.figure(figsize=figsize)

    for i, final_flavour in enumerate(final_flavours):

        ax = plt.subplot(1, n, i+1)

        pcm = ax.pcolormesh(
            E.value,
            np.cos(Thetaz),
            P_alpha[final_flavour],
            shading="auto",
            cmap="hot"
        )

        # Remove axes completely
        ax.axis("off")
        ax.set_xscale("log")

    # Optional title (still works without axes)
    if title is not None:
        fig.suptitle(title)

    # Remove all padding/margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save as PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.close(fig)


