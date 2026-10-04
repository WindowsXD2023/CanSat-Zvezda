from matplotlib import pyplot as plt
import random, numpy as np
from time import sleep


class Environment:
    def __init__(self, wind_table):
        self.g = 9.81 # m/s²
        self.rho = 1.225 # kg/m³
        self.table = wind_table

    def _wind_blow(self):
        return random.randint(1, 10), random.randint(0, 359) # (velocity, direction)

class CanSat(Environment):
    def __init__(self, weight, par_area, gps, height, size):
        super().__init__(None)
        self.mass = weight
        self.parea = par_area
        self.term_v = np.sqrt((2 * self.mass * self.g) / (self.rho * 1.5 * self.parea))
        self.height = height - self.term_v/self.g
        self.coords = gps
        self.pos = random.randint(0, 359)
        self.r, self.h = size

    def measure(self):
        self.wind_v, self.wind_d = self._wind_blow()
        return self.wind_v, self.wind_d, self.pos, self.coords, self.height

    def calculate(self, wind_vel, wind_dir):
        pos = 360-wind_dir
        force = 0.5 * self.rho * wind_vel**2 * 1.5 * self.r * np.pi * self.h
        return pos, force

    def fall(self, time):
        self.height -= time*self.term_v
        distance = time*self.wind_v
        angle = np.radians(self.wind_d)
        dx = self.coords[0] - distance*np.sin(angle)
        dy = self.coords[1] - distance*np.cos(angle)
        self.coords = (dx, dy)
        
    def log(self):
        return f"CanSat satelite log:\nRate of descent: {self.term_v:.2f}m/s\nCoordinates (x,y): {float(self.coords[0]):.4}, {float(self.coords[0]):.4}\nRotation: {self.pos}°\nHeight: {self.height:.1f}m\nWind speed: {self.wind_v}m/s\nWind direction {self.wind_d}°\n" + "-"*20

    def counter(self, force, direction):
        pass

sat = CanSat(0.3, 0.032, (0,0), 2000, (33, 115))
positions = []
while True:
    _, _, _, position, height = sat.measure()
    if height <= 0:
        break
    positions.append(position)
    sat.fall(10)
    # sleep(1)

startx = (list(zip(*positions))[0][0], list(zip(*positions))[0][-1])
endx = (list(zip(*positions))[1][0], list(zip(*positions))[1][-1])

plt.plot(*zip(*positions), startx, endx, "o--")
plt.title("CanSat position in time domain")
plt.xlabel("X coordinates")
plt.ylabel("Y coordinates")
plt.grid()
plt.show()