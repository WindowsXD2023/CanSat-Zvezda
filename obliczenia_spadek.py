# symulacja spadku ze spadochronem

import matplotlib.pyplot as plt
import numpy as np

# SPADEK BEZ SPADOCHRONU

M = 0.3 # masa satelity
G = 9.81 # przyspieszenie grawitacyjne
CD = 0.8 # współczynnik oporu powietrza
R = 0.033 # promień podstawy satelity
A = np.pi*R**2 # powierzchnia satelity
H = 2000 # wysokość spadku
AD = 1.225 # gęstość powietrza
WV = 10 # prędkość wiatru
LAT = 52.199914 # szerokość geograficzna
LONG = 21.179039 # długość geograficzna

drag_force = lambda density, velocity, coefficient, area: 0.5 * density * velocity**2 * coefficient * area # F_op = 0.5 * d * v² * C_d * A
term_velocity = lambda mass, density, coefficient, area: np.sqrt((2 * mass * G) / (density * coefficient * area)) # v_t = sqrt((2 * m * g) / (d * C_d * A))

# H = H_0 + H_1
# t = t_0 + t_1
# t_0 = v_t / g
# t_1 = H_1 / v_t
# H_0 = 0.5 * g * t_0² = 0.5 * g * v_t^2 / g² = v_t² / 2*g
# H = v_t² / 2*g + t_1 * v_t
# t = v_t / g + H_1 / v_t = v_t + (H - H_0) / v_t = v_t / g + H / v_t - v_t / 2*g

v_t = 10 # prędkość końcowa z góry ustalona przez obecnośc spadochronu
CD = 1.5
A = 2*M*G / (AD * v_t**2 * CD)
print("\n====SYMULACJA ZE SPADOCHRONEM")
print(f"Satelita ze spadochronem o powierzchni {A:.3f}m² osiągnie zakładaną prędkość krytyczną {v_t:.2f}m/s po czasie {v_t/G:.2f}s")
t = v_t / G + H / v_t - v_t / (2*G)
print(f"Spadek z wysokości {H}m zajmie mu {t:.2f}s")
print(f"W tym czasie wiatr o prędkości {WV}m/s zdryfuje satelitę o {WV*t:.1f}m")

