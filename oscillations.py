import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

# Constants ===============================================================

class OscillationParameters:
    def __init__(self, dm21=7.49e-5*u.eV**2, dm32=2.513e-3*u.eV**2, sin212=0.308, sin223=0.470, sin213=2.215e-2, deltaCP=3.7, hierarchy=1): # for mixing angles, sin2ij=sin^2(theta_ij)
        self.dm21 = dm21
        self.dm32 = dm32
        self.sin212 = sin212
        self.sin223 = sin223
        self.sin213 = sin213
        self.deltaCP = deltaCP
        self.hierarchy = hierarchy # 1 NH, -1 IH


class Constants:
    def __init__(self, Gf=1.2e-5/u.GeV**2, hc=200*u.MeV*u.fm, c=3e8*u.m/u.s, mp=938*u.MeV):
        self.Gf = Gf
        self.hc = hc
        self.c = c
        self.mp = mp/c**2


def PMNS(osc):
    """
    Returns the PMNS matrix for the given parameters.
    """
    
    theta12 = np.arcsin(np.sqrt(osc.sin212))
    theta23 = np.arcsin(np.sqrt(osc.sin223))
    theta13 = np.arcsin(np.sqrt(osc.sin213))
    deltaCP = osc.deltaCP
    
    c12 = np.cos(theta12)
    s12 = np.sin(theta12)
    c13 = np.cos(theta13)
    s13 = np.sin(theta13)
    c23 = np.cos(theta23)
    s23 = np.sin(theta23)

    U = np.array([[c12*c13, s12*c13, s13*np.exp(-1j*deltaCP)], [-s12*c23-c12*s23*s13*np.exp(1j*deltaCP), c12*c23-s12*s23*s13*np.exp(1j*deltaCP), s23*c13], [s12*s23-c12*c23*s13*np.exp(1j*deltaCP), -c12*s23-s12*c23*s13*np.exp(1j*deltaCP), c23*c13]])
    return U


def MassSquared(osc):
    """
    Returns the mass matrix for the given masses.
    """
    if osc.hierarchy == 1:
        return np.diag([0, osc.dm21.value, osc.dm32.value + osc.dm21.value])*osc.dm21.unit
    elif osc.hierarchy == -1:
        return np.diag([0, osc.dm21.value, - osc.dm32.value - osc.dm21.value])*osc.dm21.unit


# Earth utilities ==========================================================



def is_between(xA, yA, xB, yB, x1, y1):

    if x1 >= min(xA,xB) and x1 <= max(xA,xB) and y1 >= min(yA,yB) and y1 <= max(yA,yB) :
        return True

    return False


def find_intersections(xA, yA, xB, yB, radii) :

    intersections = []

    if xA == xB :
        intersections = np.array([[0, radius] for radius in radii])
        return np.array([[x,y] for [x,y] in intersections if is_between(xA, yA, xB, yB, x, y)])
    
    m = (yB - yA) / (xB - xA)

    D = np.sqrt((xB - xA)**2 + (yB - yA)**2) # Distance between A and B

    for radius in radii :

        a = 1 + m**2
        b = 2*yA*m - 2*m**2*xA
        c = (yA - m*xA)**2 - radius**2

        delta = b**2 - 4*a*c

        if delta >= 0:
            x1 = (-b + np.sqrt(delta)) / (2*a)
            x2 = (-b - np.sqrt(delta)) / (2*a)

            y1 = m*x1 + yA - m*xA
            y2 = m*x2 + yA - m*xA

            if is_between(xA, yA, xB, yB, x1, y1) :
                intersections.append([x1, y1])
            if is_between(xA, yA, xB, yB, x2, y2) :
                intersections.append([x2, y2])
        
        elif delta < 0:
            continue

    return np.array(intersections)


def find_slabs(layers, intersections, xProd, yProd, xSK, ySK) :

    D = np.sqrt((xSK - xProd)**2 + (ySK - yProd)**2)
    distances = []

    for [x, y] in intersections : #compute the distance between the intersection and the production point

        d = np.sqrt((x - xProd)**2 + (y - yProd)**2)
        distances.append(d)

    sorted_intersections = [x for _, x in sorted(zip(distances, intersections))] #sort the intersections by distance to production point
    intersection_prod_detec = np.zeros((len(sorted_intersections)+2, 2))
    intersection_prod_detec[0] = [xProd, yProd]
    intersection_prod_detec[-1] = [xSK, ySK]
    intersection_prod_detec[1:-1] = sorted_intersections

    slabs = [{'entry': entry, 'exit': exit} for entry, exit in zip(intersection_prod_detec, intersection_prod_detec[1:])]
    
    for slab in slabs :

        xentry, yentry = slab['entry']
        xexit, yexit = slab['exit']
        d = np.sqrt((xexit- xentry)**2 + (yexit - yentry)**2)
        slab['thickness'] = d

        xmid, ymid = (xentry + xexit) / 2, (yentry + yexit) / 2
        Rmid = np.sqrt(xmid**2 + ymid**2)

        layer_key = None
        for key, value in layers.items():
            if value['Rmin'] <= Rmid <= value['Rmax']:
                layer_key = key
                break
        slab['layer'] = layer_key
        

    return slabs


# Oscillations utilities ==================================================


def evolution_through_matter(L, E, rho, Z_over_A, osc, constants, anti=False) : # evolution matrix through matter of the probability amplitude psi_\alpha\beta

    UPMNS = PMNS(osc)

    M2 = MassSquared(osc)

    Ne = Z_over_A*(constants.hc)**3/constants.mp*rho # convert rho to electron density in natural unit
    Acc = 2*E*np.sqrt(2)*constants.Gf*Ne

    if anti :
        Acc = -Acc
        UPMNS = np.conjugate(UPMNS)

    # convert L to natural units

    Lnat = L/constants.hc

    HF = 0.5/E*(UPMNS @ M2 @ np.transpose(np.conjugate(UPMNS)) + np.diag([Acc.value, 0., 0.])*Acc.unit)

    eigenvalues, eigenvectors = np.linalg.eigh(HF)

    UM = eigenvectors

    diag = np.exp(-1.0j*eigenvalues*Lnat)
    evol = np.diag(diag.value)

    return UM @ evol @ np.transpose(np.conjugate(UM))


def proba_amp(L, E, rho, Z_over_A, osc, constants, psi0, anti=False) : # proba amplutude psi_\alpha\beta through matter

    U = evolution_through_matter(L, E, rho, Z_over_A, osc, constants, anti)
    psi = U @ psi0

    return psi


def evolution_through_matter_LBL(L, E, rho, osc, constants, anti=False) : # evolution matrix through matter of the probability amplitude psi_\alpha\beta

    dm21 = osc.dm21
    dm32 = osc.dm32
    dm31 = dm32 + dm21

    dm31 = osc.hierarchy*dm31

    Ne = 0.1*(constants.hc)**3/constants.mp*rho # convert rho to electron density in natural unit
    Acc = 2*E*np.sqrt(2)*constants.Gf*Ne

    if anti :
        Acc = -Acc

    # convert L to natural units

    Lnat = L/constants.hc

    theta13 = np.arcsin(np.sqrt(osc.sin213))
    theta23 = np.arcsin(np.sqrt(osc.sin223))
    
    Dm2_13M = np.sqrt((dm31*np.cos(2*theta13) - Acc)**2 + (dm31*np.sin(2*theta13))**2)
    #theta13_M = 0.5*np.arctan(np.tan(2*theta13)/(1 - Acc/(dm31*np.cos(2*theta13))))
    Gamma = Acc/(dm31)
    theta13_M = 0.5*np.arcsin(np.sin(2*theta13)**2/(np.sin(2*theta13)**2 + (Gamma - np.cos(2*theta13))**2))

    R13_M = np.array([[np.cos(theta13_M), 0, np.sin(theta13_M)], [0, 1, 0], [-np.sin(theta13_M), 0, np.cos(theta13_M)]])
    R23 = np.array([[1, 0, 0], [0, np.cos(theta23), np.sin(theta23)], [0, -np.sin(theta23), np.cos(theta23)]])
    U_M = R23 @ R13_M

    Dm_1M = dm31 + Acc - Dm2_13M
    Dm_3M = dm31 + Acc + Dm2_13M

    diag = np.array([np.exp(-0.25j*Dm_1M*Lnat/E), 1, np.exp(-0.25j*Dm_3M*Lnat/E)])
    evol = np.diag(diag)

    return U_M @ evol @ np.transpose(np.conjugate(U_M))


def proba_amp_LBL(L, E, rho, osc, constants, psi0, anti=False) : # proba amplutude psi_\alpha\beta through matter

    U = evolution_through_matter_LBL(L, E, rho, osc, constants, anti)
    psi = U @ psi0

    return psi


def proba_through_earth(thetaz, E, layers, radii, RP, RSK, psi0, osc, constants, anti=False, plot_earth=False, fig=None, ax=None) :

    # find production point for a given zenith angle

    xA, yA = 0, RSK

    if thetaz == 0:
        xB, yB = 0, RP
    else :
        xB, yB = RP + 100, RSK + (RP + 100)/np.tan(thetaz)

    XP, YP = find_intersections(xA, yA, xB, yB, [RP])[0]
    XSK, YSK = 0, RSK

    intersections = find_intersections(XP, YP, XSK, YSK, radii)
    slabs = find_slabs(layers, intersections, XP, YP, XSK, YSK)

    # debug plot

    if plot_earth :

        # intersections
        for slab in slabs :
            plt.plot([slab['entry'][0], slab['exit'][0]], [slab['entry'][1], slab['exit'][1]], color=layers[slab['layer']]['Color'], linestyle='-', zorder=10)

        plt.scatter(intersections[:,0], intersections[:,1], color='black', marker='+', zorder=10)
        plt.scatter(XP, YP, color='b', marker='x', zorder=10)


    psi = psi0

    for slab in slabs:
        
        layer = layers[slab['layer']]
        rho = layer['Density']

        L = slab['thickness']*u.km
        Z_over_A = layer['Z/A']
        psi = proba_amp(L, E, rho, Z_over_A, osc, constants, psi, anti=anti)


    return psi


def plot_earth(layers, radii, RP, RSK) :

    fig, ax = plt.subplots(figsize=(6,6))

    circle1 = plt.Circle((0, 0), RSK, color='lightblue', zorder=0)
    ax.add_artist(circle1)

    for key, layer in layers.items() :

        circle = plt.Circle((0, 0), layer['Rmax'], color=layer['Color'], zorder=1)
        ax.add_artist(circle)

    ax.set_xlim(-RP-500, RP+500)
    ax.set_ylim(-RP-500, RP+500)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title('Earth model')

    plt.show()

    return fig, ax


def plot_proba(E, thetaz, P_mu_mu, P_mu_e, title="", figsize=(17, 7)) :

    plt.figure(figsize=figsize)

    # plot P(nu_mu -> nu_mu)

    plt.subplot(1, 2, 1)

    vmin = 0
    vmax = np.max(P_mu_mu)

    plt.pcolormesh(E, np.cos(thetaz), P_mu_mu, cmap='hot_r', vmin=vmin, vmax=vmax, rasterized=True)
    plt.title(r'$P\left(\nu_\mu \rightarrow \nu_\mu\right)$')
    plt.xlabel(r'$E_\nu$ (GeV)')
    plt.ylabel(r'$\cos\theta_z$')
    plt.xscale('log')
    plt.colorbar(label='Probability')

    # plot P(nu_mu -> nu_e)

    plt.subplot(1, 2, 2)

    vmin = 0
    vmax = np.max(P_mu_e)

    plt.pcolormesh(E, np.cos(thetaz), P_mu_e, cmap='hot_r', vmin=vmin, vmax=vmax, rasterized=True)
    plt.title(r'$P\left(\nu_\mu \rightarrow \nu_e\right)$')
    plt.xlabel(r'$E_\nu$ (GeV)')
    plt.ylabel(r'$\cos\theta_z$')
    plt.xscale('log')
    plt.colorbar(label='Probability')

    plt.suptitle(title)

    plt.tight_layout()





