import astropy.units as u
class Constants:
    def __init__(self, Gf=1.2e-5/u.GeV**2, hc=200*u.MeV*u.fm, c=3e8*u.m/u.s, mp=938*u.MeV):
        self.Gf = Gf
        self.hc = hc
        self.c = c
        self.mp = mp/c**2

GF = 1.2e-5/u.GeV**2 # Fermi coupling constant
HC = 200*u.MeV*u.fm # hc in MeV*fm, useful for converting between energy and length scales
C = 3e8*u.m/u.s # speed of light in m/s
MP = 938*u.MeV/C**2 # proton mass


