import numpy as np
import astropy.units as u

import torch

import pyoscprob.const as const

from tqdm import tqdm


FLAVOR_TO_INDEX = {
    "e": 0,
    "electron": 0,
    "mu": 1,
    "muon": 1,
    "tau": 2
}

INDEX_TO_FLAVOR = {
    0: "e",
    1: "mu",
    2: "tau"
}

def flavor_to_index(flavor):

    if isinstance(flavor, int):
        return flavor

    if flavor.lower() in FLAVOR_TO_INDEX:
        return FLAVOR_TO_INDEX[flavor.lower()]

    raise ValueError(f"Unknown flavor: {flavor}")


class Oscillator:

    def __init__(self, osc_params, device="cpu"):
        """
        osc_params : OscillationParameters
        constants  : constants container
        earth_model: EarthModel (optional)
        """

        self.osc = osc_params



        self.device = device

    # --------------------------------------------------
    # Evolution operator through matter
    # --------------------------------------------------

    def evolution_operator(self, L, E, rho, Z_over_A, anti=False):

        UPMNS = self.osc.PMNS
        M2 = self.osc.M2

        # electron density
        Ne = Z_over_A * (const.HC)**3 / const.MP * rho

        Acc = 2 * E * np.sqrt(2) * const.GF * Ne

        if anti:
            Acc = -Acc
            UPMNS = np.conjugate(UPMNS)


        # convert baseline
        Lnat = L / const.HC

        HF = 0.5 / E * (
            UPMNS @ M2 @ np.transpose(np.conjugate(UPMNS))
            + np.diag([Acc.value, 0.0, 0.0]) * Acc.unit
        )

        eigenvalues, eigenvectors = np.linalg.eigh(HF)


        UM = eigenvectors

        diag = np.exp(-1j * eigenvalues * Lnat)
        evol = np.diag(diag.value)

        return UM @ evol @ np.transpose(np.conjugate(UM))
    

    def evolution_operator_vectorized(self, L, E_array, rho, Z_over_A, anti=False):
        """
        Vectorized evolution operator using PyTorch.
        All inputs are Astropy Quantities, but internally converted to floats.
        Returns U_all: (N_E,3,3) torch tensor.
        """
        # Convert energies to eV and lengths to natural units
        E = torch.tensor(E_array.to(u.GeV).value, dtype=torch.cdouble, device=self.device)  # (N_E,)
        

        # convert to natural units
        L_nat = L / const.HC
        Ne_nat = Z_over_A * rho / const.MP * (const.HC)**3  
        # convert to GeV and ditch units for the Hamiltonian
        L_nat = L_nat.to(u.GeV**(-1)).value
        Ne_nat = Ne_nat.to(u.GeV**3).value
        GF = const.GF.to(1/u.GeV**2).value

        Hvac = self.osc.PMNS @ self.osc.M2 @ self.osc.PMNS.conj().T
        Hvac = Hvac.to(u.GeV**2).value

        # matter potential in eV
        Vcc = np.sqrt(2) * GF * Ne_nat  # GeV

        if anti:
            Vcc = -Vcc
            Hvac = Hvac.T


        # Vacuum Hamiltonian as torch tensor in eV
        Hvac = torch.tensor(Hvac, dtype=torch.cdouble, device=self.device)

        # Build batch Hamiltonians: H_batch[i] = Hvac/(2E_i) + diag(Vcc,0,0)/(2E_i)
        H_batch = Hvac[None, :, :] / (2 * E[:, None, None])
        H_batch[:, 0, 0] += Vcc  # electron flavor only

        # Batched eigen-decomposition
        eigvals, eigvecs = torch.linalg.eigh(H_batch)  # shapes: (N_E,3),(N_E,3,3)

        # Evolution operator: U = V diag(exp(-i λ L)) V†
        phases = torch.exp(-1j * eigvals * L_nat)
        diag_phase = torch.diag_embed(phases)
        U_all = torch.matmul(eigvecs, torch.matmul(diag_phase, eigvecs.conj().transpose(-2, -1)))

        return U_all

    # --------------------------------------------------
    # propagate amplitude
    # --------------------------------------------------

    def propagate_amplitude(self, psi0, L, E, rho, Z_over_A, anti=False):

        U = self.evolution_operator(L, E, rho, Z_over_A, anti)

        return U @ psi0

    # --------------------------------------------------
    # propagate through slabs
    # --------------------------------------------------

    def propagate(self, psi0, slabs, E, anti=False):

        psi = psi0

        for slab in slabs:

            L = slab.thickness
            rho = slab.layer.density
            Z_over_A = slab.layer.Z_over_A

            U = self.evolution_operator(L, E, rho, Z_over_A, anti)

            psi = U @ psi

        return psi
    
    def propagate_vectorized(self, psi0, slabs, E_array, anti=False):
        N_E = len(E_array)
        psi = torch.tensor(psi0, dtype=torch.cdouble, device=self.device)[:, None].repeat(1, N_E)

        for slab in slabs:
            L = slab.thickness
            rho = slab.layer.density
            Z_over_A = slab.layer.Z_over_A

            U_all = self.evolution_operator_vectorized(L, E_array, rho, Z_over_A, anti)
            psi = torch.bmm(U_all, psi.T.unsqueeze(-1)).squeeze(-1)
            psi = psi.T

        return psi

    # --------------------------------------------------
    # probability
    # --------------------------------------------------

    def probabilities_single(self, alpha, slabs, E, anti=False):
        """
        Compute probabilities for a single energy.
        """
        alpha = flavor_to_index(alpha)
        psi0 = np.zeros(3, dtype=complex)
        psi0[alpha] = 1

        psi = self.propagate(psi0, slabs, E, anti)
        probs = np.abs(psi)**2

        return {"e": probs[0], "mu": probs[1], "tau": probs[2]}
    

    def probabilities_vectorized(self, alpha, slabs, E_array, anti=False):
        alpha_idx = {"e":0, "mu":1, "tau":2}[alpha] if isinstance(alpha,str) else alpha
        psi0 = np.zeros(3, dtype=complex)
        psi0[alpha_idx] = 1.0

        psi = self.propagate_vectorized(psi0, slabs, E_array, anti)

        probs = np.abs(psi.cpu().numpy())**2  # convert back to numpy
        return {"e": probs[0], "mu": probs[1], "tau": probs[2]}
    


    def probabilities(self, alpha, slabs, E, anti=False):
        """
        Automatically dispatch to single-energy or array-energy calculation.
        """

        # Astropy Quantity
        if isinstance(E, u.Quantity):

            if E.isscalar:
                return self.probabilities_single(alpha, slabs, E, anti)
            else:
                return self.probabilities_vectorized(alpha, slabs, E, anti)

        else:
            raise ValueError("Energy must be an astropy Quantity")