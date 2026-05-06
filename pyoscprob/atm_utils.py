import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from tqdm import tqdm

import os
import uproot

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

flavour_to_nu = {
    "e": "nue",
    "mu": "numu",
    "tau": "nutau",
}

flavour_to_nu_anti = {
    "e": "nuebar",
    "mu": "numubar",
    "tau": "nutaubar",
}


def compute_oscillograms(alpha, Thetaz, E, earth, osc_params, anti=False, prod_height_file=None, verbose=1):
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
    prod_height_file : str, optional
        Path to file containing production heights for atmospheric neutrinos. If None, a default height of 15 km is used for all neutrinos. If not None, will compute smeared probabilities (default: None).
    smear : bool, optional
        Whether to apply smearing to the production heights (default: False).

    Returns
    -------
    P_alpha : dict
        Dictionary with keys "e", "mu", "tau", each an array of shape
        (len(Thetaz), len(E)), giving the probabilities for that final flavor.
    """

    # Precompute slabs for all theta values
    if verbose: print("Precomputing slabs for all thetas...")
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
    if verbose: 
        print("Computing oscillation probabilities...")
        for i in tqdm(range(N_theta), total=N_theta):
            
            probs = osc.probabilities(alpha, Slabs_list[i], E, anti=anti)

            # Fill arrays
            for flavor in ["e", "mu", "tau"]:
                P_alpha[flavor][i] = probs[flavor]

            
    else:
        for i in range(N_theta):
            
            probs = osc.probabilities(alpha, Slabs_list[i], E, anti=anti)

            # Fill arrays
            for flavor in ["e", "mu", "tau"]:
                P_alpha[flavor][i] = probs[flavor]


    if verbose: print("Done.")

    return P_alpha


def smeared_probabilities(alpha, Thetaz, E, osc_params, layers=None, anti=False, prod_height_file=None):

    # Open production height file if provided
    print(f"Loading production heights from '{prod_height_file}'...")

    if not os.path.isfile(prod_height_file):
        raise FileNotFoundError(f"Production height file '{prod_height_file}' not found.")

    prod_heights = uproot.open(prod_height_file)
    flav = flavour_to_nu_anti[alpha] if anti else flavour_to_nu[alpha]
    height_histo = prod_heights[f"hprodheight_{flav}"]
    edges_E = height_histo.axis(0).edges()
    edges_tz = height_histo.axis(1).edges()
    edges_h = height_histo.axis(2).edges()
    height_values = height_histo.values()
    # sum over E to get distribution of production height vs cos(theta_z)
    height_values = np.sum(height_values, axis=0)
    # normalize to get probability distribution per cos(theta_z)
    height_values /= np.sum(height_values, axis=1, keepdims=True)

    # for each theta_z, sample a production height from the distribution and compute probabilities
    P_alpha_smeared = {
        "e": np.zeros((len(Thetaz), len(E))),
        "mu": np.zeros((len(Thetaz), len(E))),
        "tau": np.zeros((len(Thetaz), len(E))),
    }

    # Loop over zenith angles
    print("Computing smeared probabilities...")
    for i, thetaz in tqdm(enumerate(Thetaz), total=len(Thetaz)):

        # find closest bin in edges_tz
        tz_idx = np.searchsorted(edges_tz, thetaz) - 1
        tz_idx = np.clip(tz_idx, 0, len(edges_tz)-2)

        # sample production height for this theta_z
        h_samples = np.random.choice(edges_h[:-1], size=100, p=height_values[tz_idx])

        # clip to min height of 0 km and max height of 100 km
        h_samples = np.clip(h_samples, 0, 100)

        # compute probabilities for each sampled height and average
        P_alpha_height = {
            "e": np.zeros((len(h_samples), len(E))),
            "mu": np.zeros((len(h_samples), len(E))),
            "tau": np.zeros((len(h_samples), len(E))),
        }

        for j, h in enumerate(h_samples):
            earth = EarthModel(layers=layers, Rprod=6371*u.km+h*u.km)
            P_alpha_height_j = compute_oscillograms(alpha, [thetaz], E, earth, osc_params, anti=anti, verbose=0)
            for flavor in ["e", "mu", "tau"]:
                P_alpha_height[flavor][j] = P_alpha_height_j[flavor][0]

        for flavor in ["e", "mu", "tau"]:
            P_alpha_smeared[flavor][i] = np.mean(P_alpha_height[flavor], axis=0)

    return P_alpha_smeared


def smeared_probabilities_fast(alpha, Thetaz, E, osc_params, layers=None,
                               anti=False, prod_height_file=None,
                               n_samples=50):

    print(f"Loading production heights from '{prod_height_file}'...")

    if not os.path.isfile(prod_height_file):
        raise FileNotFoundError(f"Production height file '{prod_height_file}' not found.")

    prod_heights = uproot.open(prod_height_file)

    flav = flavour_to_nu_anti[alpha] if anti else flavour_to_nu[alpha]
    height_histo = prod_heights[f"hprodheight_{flav}"]

    edges_tz = height_histo.axis(1).edges()
    edges_h = height_histo.axis(2).edges()

    height_values = height_histo.values()
    height_values = np.sum(height_values, axis=0)
    height_values /= np.sum(height_values, axis=1, keepdims=True)

    osc = Oscillator(osc_params)

    P_alpha_smeared = {
        "e": np.zeros((len(Thetaz), len(E))),
        "mu": np.zeros((len(Thetaz), len(E))),
        "tau": np.zeros((len(Thetaz), len(E))),
    }

    print("Computing smeared probabilities (fast)...")

    for i, thetaz in tqdm(enumerate(Thetaz), total=len(Thetaz)):

        # Find theta bin
        tz_idx = np.searchsorted(edges_tz, thetaz) - 1
        tz_idx = np.clip(tz_idx, 0, len(edges_tz)-2)

        # Sample heights
        h_samples = np.random.choice(edges_h[:-1], size=n_samples,
                                     p=height_values[tz_idx])
        h_samples = np.clip(h_samples, 0, 100)

        # Accumulate probabilities
        P_tmp = {"e": 0, "mu": 0, "tau": 0}

        for h in h_samples:
            earth = EarthModel(layers=layers,
                               Rprod=6371*u.km + h*u.km)

            slabs = earth.compute_slabs(thetaz)
            probs = osc.probabilities(alpha, slabs, E, anti=anti)

            for f in P_tmp:
                P_tmp[f] += probs[f]

        # Average
        for f in P_tmp:
            P_alpha_smeared[f][i] = P_tmp[f] / n_samples

    return P_alpha_smeared


def smeared_probabilities_weighted(alpha, Thetaz, E, osc_params,
                                   layers=None, anti=False,
                                   prod_height_file=None,
                                   prob_threshold=1e-4):

    print(f"Loading production heights from '{prod_height_file}'...")

    if not os.path.isfile(prod_height_file):
        raise FileNotFoundError(f"Production height file '{prod_height_file}' not found.")

    prod_heights = uproot.open(prod_height_file)

    flav = flavour_to_nu_anti[alpha] if anti else flavour_to_nu[alpha]
    height_histo = prod_heights[f"hprodheight_{flav}"]

    edges_tz = height_histo.axis(1).edges()
    edges_h = height_histo.axis(2).edges()

    # Bin centers (IMPORTANT)
    h_centers = 0.5 * (edges_h[:-1] + edges_h[1:])

    height_values = height_histo.values()
    height_values = np.sum(height_values, axis=0)

    # Normalize per theta bin
    height_values /= np.sum(height_values, axis=1, keepdims=True)

    osc = Oscillator(osc_params)

    P_alpha = {
        "e": np.zeros((len(Thetaz), len(E))),
        "mu": np.zeros((len(Thetaz), len(E))),
        "tau": np.zeros((len(Thetaz), len(E))),
    }

    print("Computing smeared probabilities (weighted)...")

    for i, thetaz in tqdm(enumerate(Thetaz), total=len(Thetaz)):

        tz_idx = np.searchsorted(edges_tz, thetaz) - 1
        tz_idx = np.clip(tz_idx, 0, len(edges_tz)-2)

        weights = height_values[tz_idx]

        # Optional: skip negligible bins
        mask = weights > prob_threshold
        h_used = h_centers[mask]
        w_used = weights[mask]

        # Renormalize after cut
        w_used /= np.sum(w_used)

        P_tmp = {"e": 0, "mu": 0, "tau": 0}

        for h, w in zip(h_used, w_used):

            earth = EarthModel(
                layers=layers,
                Rprod=6371*u.km + h*u.km
            )

            slabs = earth.compute_slabs(thetaz)
            probs = osc.probabilities(alpha, slabs, E, anti=anti)

            for f in P_tmp:
                P_tmp[f] += w * probs[f]

        for f in P_tmp:
            P_alpha[f][i] = P_tmp[f]

    return P_alpha


def build_energy_smearing_matrix(E):
    """
    Build Gaussian smearing matrix:
    S[i, j] = P(E_reco_i | E_true_j)
    """
    E_vals = E.value
    nE = len(E_vals)

    S = np.zeros((nE, nE))

    for j, E_true in enumerate(E_vals):
        sigma = 0.02 * E_true

        # Gaussian centered at E_true
        S[:, j] = np.exp(-(E_vals - E_true)**2 / (2 * sigma**2))

        # Normalize column (important!)
        S[:, j] /= np.sum(S[:, j])

    return S


def plot_oscillograms(Thetaz, E, P_alpha, init_flavour, final_flavours=["e", "mu", "tau"], anti=False, title=None):

    if len(final_flavours) == 0:
        raise ValueError("At least one final flavor must be specified.")
    elif len(final_flavours) > 3:
        raise ValueError("At most three final flavors can be specified.")
    elif len(final_flavours) == 1:
        #figsize = (12, 10)
        figsize=(8, 6)
    elif len(final_flavours) == 2:
        figsize = (12, 5)
    elif len(final_flavours) == 3:
        figsize = (18, 5)

    plt.figure(figsize=figsize)

    for i, final_flavour in enumerate(final_flavours):

        plt.subplot(1, len(final_flavours), i+1)

        plt.pcolormesh(E.value, np.cos(Thetaz), P_alpha[final_flavour], shading="auto", cmap="hot", vmin=0, vmax=1)
        plt.colorbar(label="Probability")

        plt.xscale("log")
        plt.xlabel(r"$E_\nu$" + f" [{E.unit}]")
        plt.ylabel(r"cos($\theta_z$)")

        if anti:
            plt.title(f"${flavour_to_latex_anti[init_flavour]} \\to {flavour_to_latex_anti[final_flavour]}$")
        else:
            plt.title(f"${flavour_to_latex[init_flavour]} \\to {flavour_to_latex[final_flavour]}$")

    if title is not None:
        plt.suptitle(title, fontsize=25)

    plt.tight_layout()
    #plt.show()

def plot_oscillograms_noaxes(Thetaz, E, P_alpha, init_flavour,
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


