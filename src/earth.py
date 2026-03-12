import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from dataclasses import dataclass


# --------------------------------------------------
# Layer class
# --------------------------------------------------

@dataclass
class Layer:
    name: str
    Rmin: float
    Rmax: float
    density: float
    Z_over_A: float

    def contains(self, r):
        return self.Rmin <= r <= self.Rmax


# --------------------------------------------------
# Slab class
# --------------------------------------------------

@dataclass
class Slab:
    entry: np.ndarray
    exit: np.ndarray
    thickness: float
    layer: Layer


# --------------------------------------------------
# Earth model
# --------------------------------------------------

class EarthModel:

    def __init__(self, layers=None, Rdet=6370*u.km, Rprod=6371*u.km + 15*u.km):

        if layers is None:
            self.layers = [
                Layer("Atmosphere", 6371*u.km, (6371 + 500)*u.km, 0.0*u.g/u.cm**3, 0.4),
                Layer("Crust", 5701*u.km, 6371*u.km, 3.3*u.g/u.cm**3, 0.4),
                Layer("Mantle", 3480*u.km, 5701*u.km, 5.0*u.g/u.cm**3, 0.4),
                Layer("Outer Core", 1220*u.km, 3480*u.km, 11.3*u.g/u.cm**3, 0.4),
                Layer("Inner Core", 0*u.km, 1220*u.km, 13.0*u.g/u.cm**3, 0.4),
            ]
        else:
            self.layers = layers

        self.Rdet = Rdet # distance from Earth's center to detector
        self.Rprod = Rprod # distance from Earth's center to production point in the atmosphere

        # radii defining boundaries
        self.radii = [layer.Rmax for layer in self.layers]

    # --------------------------------------------------

    @staticmethod
    def _is_between(xA, yA, xB, yB, x1, y1):

        return (
            min(xA, xB) <= x1 <= max(xA, xB)
            and min(yA, yB) <= y1 <= max(yA, yB)
        )

    # --------------------------------------------------

    def _find_intersections(self, xA, yA, xB, yB, radii):

        intersections = []

        # vertical line case
        if np.isclose(xA, xB):

            intersections = np.concatenate([np.array([[xA, -r.value] for r in radii]), np.array([[xA, r.value] for r in radii])])

            return np.array([
                [x, y]
                for (x, y) in intersections
                if self._is_between(xA, yA, xB, yB, x, y)
            ])

        m = (yB - yA) / (xB - xA)

        for radius in radii:
            radius = radius.value

            a = 1 + m**2
            b = 2 * yA * m - 2 * m**2 * xA
            c = (yA - m * xA)**2 - radius**2

            delta = b**2 - 4 * a * c

            if delta >= 0:

                x1 = (-b + np.sqrt(delta)) / (2 * a)
                x2 = (-b - np.sqrt(delta)) / (2 * a)

                y1 = m * x1 + yA - m * xA
                y2 = m * x2 + yA - m * xA

                if self._is_between(xA, yA, xB, yB, x1, y1):
                    intersections.append([x1, y1])

                if self._is_between(xA, yA, xB, yB, x2, y2):
                    intersections.append([x2, y2])

        return np.array(intersections)

    # --------------------------------------------------

    def _find_layer(self, r):

        for layer in self.layers:
            if layer.contains(r):
                return layer

        return None
    

    # --------------------------------------------------

    def production_point(self, thetaz, ax=None): # find production point in the atmosphere for a given zenith angle

        xDet, yDet = 0, self.Rdet.value

        theta = float(thetaz)
        if np.isclose(theta, 0.):
            return 0, self.Rprod.value
        elif np.isclose(theta, np.pi):
            return 0, -self.Rprod.value

        xFar = self.Rprod.value + 100
        yFar = self.Rdet.value + (self.Rprod.value + 100)/np.tan(theta)

        if ax is not None:
            ax.scatter(xFar, yFar, color="blue")

        intersections = self._find_intersections(
            xDet, yDet, xFar, yFar, radii=[self.Rprod]
        )

        if len(intersections) == 0:
            raise RuntimeError("No production intersection found")

        # choose the one farthest from detector
        distances = [
            np.sqrt((x-xDet)**2 + (y-yDet)**2)
            for x,y in intersections
        ]

        return intersections[np.argmax(distances)]
    

    def compute_slabs(self, thetaz):

        xProd, yProd = self.production_point(thetaz)
        xDet, yDet = 0, self.Rdet.value


        intersections = self._find_intersections(
            xProd, yProd, xDet, yDet, radii=self.radii
        )

        distances = []

        for x, y in intersections:
            d = np.sqrt((x - xProd)**2 + (y - yProd)**2)
            distances.append(d)

        # sort intersections by distance to production point
        idx = np.argsort(distances)
        sorted_intersections = intersections[idx]
        

        points = np.zeros((len(sorted_intersections) + 2, 2))

        points[0] = [xProd, yProd]
        points[-1] = [xDet, yDet]

        if len(sorted_intersections) > 0:
            points[1:-1] = sorted_intersections

        slabs = []

        for entry, exit in zip(points, points[1:]):

            xentry, yentry = entry
            xexit, yexit = exit

            thickness = np.sqrt(
                (xexit - xentry)**2 + (yexit - yentry)**2
            )

            xmid = (xentry + xexit) / 2
            ymid = (yentry + yexit) / 2

            Rmid = np.sqrt(xmid**2 + ymid**2)*self.Rdet.unit
            layer = self._find_layer(Rmid)

            slab = Slab(
                entry=np.array(entry)*self.Rdet.unit,
                exit=np.array(exit)*self.Rdet.unit,
                thickness=thickness*self.Rdet.unit,
                layer=layer
            )

            slabs.append(slab)

        return slabs
    

# ------------------- debug plot code -------------------

    def plot_slabs(self,Thetaz):

        Slabs = [self.compute_slabs(thetaz) for thetaz in Thetaz]

        colors = {"Atmosphere": "lightblue", "Crust": "green", "Mantle": "sienna", "Outer Core": "chocolate", "Inner Core": "firebrick"} # colors for the layers

        fig, ax = plt.subplots(figsize=(7,7))

        # --------------------------------
        # Draw Earth layers
        # --------------------------------
        for layer in self.layers:

            circle = plt.Circle(
                (0,0),
                layer.Rmax.value,
                fill=True,
                linestyle="--",
                color=colors[layer.name],
                alpha=0.5
            )

            ax.add_patch(circle)

        # --------------------------------
        # Detector
        # --------------------------------
        xDet, yDet = 0, self.Rdet.value
        ax.scatter(xDet, yDet, s=80, color="black", label="SK", marker="x")

        # --------------------------------
        # Loop over trajectories
        # --------------------------------
        for thetaz, slabs in zip(Thetaz, Slabs):

            xProd, yProd = self.production_point(thetaz)


            # intersection points
            for slab in slabs:

                x1, y1 = slab.entry.value
                x2, y2 = slab.exit.value

                ax.scatter(x1, y1, s=5, color="black")
                ax.scatter(x2, y2, s=5, color="black")

                ax.plot([x1, x2], [y1, y2], color=colors[slab.layer.name], alpha=1, zorder=1)

        # --------------------------------
        # Formatting
        # --------------------------------
        #ax.set_aspect("equal")

        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")

        ax.set_title("Atmospheric neutrino trajectories")

        #ax.grid()


        plt.show()