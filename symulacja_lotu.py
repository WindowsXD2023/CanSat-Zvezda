from matplotlib import pyplot as plt
import random, numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pogoda_api import WeatherScraper


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

    def measure(self, wind_velocity, wind_direction ):
        self.wind_v, self.wind_d = wind_velocity, wind_direction
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

scraper = WeatherScraper(51.3842, 15.1789)
scraper.fetch()
dane_pogodowe = scraper.interpolate_to_heights(target_heights_m=np.arange(0, 2501, 50))
heights = dane_pogodowe["target_heights_m"]

pogoda = []
for h in heights:
    wind_speed = dane_pogodowe[f"wind_speed_{h}m"][0]      # First timestamp
    wind_dir = dane_pogodowe[f"wind_direction_{h}m"][0]   # First timestamp
    pogoda.append((float(wind_speed), float(wind_dir)))

sat = CanSat(0.3, 0.032, (0,0), 2000, (33, 115))
positions = []
heights = []
_, _, _, start_coords, start_height = sat.measure(pogoda[0][0], pogoda[0][1])
end_coords, end_height = None, None

i = 1
while True:
    _, _, _, position, height = sat.measure(pogoda[i][0], pogoda[i][1])
    heights.append(height)
    positions.append(position)
    if height <= 0:
        _, _, _, end_coords, end_height = sat.measure(pogoda[i][0], pogoda[i][1])
        break
    sat.fall(5)
    i += 1

fig, ax = plt.subplots(figsize=(10, 8))

norm = Normalize(vmin=min(heights), vmax=max(heights))
cmap = plt.cm.viridis

sc = ax.scatter([p[0] for p in positions], [p[1] for p in positions], 
                c=heights, cmap=cmap, norm=norm, s=60, edgecolors='white', linewidth=0.5)

ax.scatter([start_coords[0]], [start_coords[1]], c='green', s=200, marker='^', 
           edgecolors='black', linewidth=1.5, label='Start', zorder=5)

if end_coords:
    ax.scatter([end_coords[0]], [end_coords[1]], c='red', s=200, marker='o', 
               edgecolors='black', linewidth=1.5, label='End', zorder=5)

ax.plot([p[0] for p in positions], [p[1] for p in positions], 
        '--', alpha=0.3, color='black', linewidth=1)

cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label('Height (m)', fontsize=10)

ax.set_title("CanSat Trajectory (colored by height)", fontsize=12, weight='bold')
ax.set_xlabel("X Coordinates", fontsize=10)
ax.set_ylabel("Y Coordinates", fontsize=10)
ax.grid(True, alpha=0.3)
ax.legend(loc='best')
plt.tight_layout()
plt.show()

