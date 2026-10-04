import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QComboBox, QGroupBox, QGridLayout, QScrollArea, QCheckBox,
)
from PyQt5.QtCore import Qt

# Stałe fizyczne
RHO = 1.225  # Gęstość powietrza na poziomie morza [kg/m³]
G = 9.81     # Przyspieszenie ziemskie [m/s²]

# Baza kształtów spadochronu i ich Cd
# Wartości Cd są orientacyjne. Dla konkretnego spadochronu najlepiej wyznaczyć je z prób.
SHAPES = {
    "Krzyżowy (Cross) - Cd = 0.75": 0.75,
    "Płaski (Parasheet) - Cd = 0.85": 0.85,
    "Stożkowy (Conical) - Cd = 0.95": 0.95,
    "Półsferyczny (Dome) - Cd = 1.30": 1.30
}


def simulate_descent(mass, area, drag_coefficient, release_height,
                     thrust, thrust_duration, wind_speed, wind_from_deg,
                     heading_deg, time_step=0.05):
    """Symuluje opadanie jako punkt masy z oporem czaszy i poziomym ciągiem.

    Kierunek wiatru oznacza, skąd wieje, a kierunek lotu to azymut śmigieł:
    0° = północ, 90° = wschód. Wynik zawiera trajektorię, prędkości i kąt czaszy.
    """
    # Zamiana kątów z kompasu na składowe wschód/północ.
    wind_angle = math.radians(wind_from_deg)
    heading = math.radians(heading_deg)
    wind_east = -wind_speed * math.sin(wind_angle)
    wind_north = -wind_speed * math.cos(wind_angle)

    # Początkowe położenie i prędkość: startujemy nad tym samym punktem na mapie.
    east = 0.0
    north = 0.0
    altitude = float(release_height)
    velocity_east = 0.0
    velocity_north = 0.0
    velocity_up = 0.0
    canopy_tilt_rad = 0.0
    canopy_lag_seconds = 1.2

    elapsed = 0.0
    maximum_tilt_deg = 0.0
    history = {
        "time": [elapsed], "east": [east], "north": [north],
        "altitude": [altitude], "speed_horizontal": [0.0],
        "speed_vertical": [0.0], "tilt_deg": [0.0], "tilt_direction": [0.0],
    }

    # Krok czasowy 0,05 s daje płynny tor bez długiego oczekiwania na wynik.
    while altitude > 0 and elapsed < 600:
        # Gęstość powietrza maleje wraz z wysokością (proste przybliżenie atmosfery).
        air_density = RHO * math.exp(-max(altitude, 0) / 8500.0)

        # Opór zależy od prędkości CanSata względem powietrza, nie względem ziemi.
        relative_east = velocity_east - wind_east
        relative_north = velocity_north - wind_north
        relative_up = velocity_up
        relative_speed = math.sqrt(
            relative_east**2 + relative_north**2 + relative_up**2
        )

        # Kwadratowy opór powietrza działa przeciwnie do ruchu względem powietrza.
        drag_factor = 0.5 * air_density * drag_coefficient * area * relative_speed
        drag_east = -drag_factor * relative_east
        drag_north = -drag_factor * relative_north
        drag_up = -drag_factor * relative_up

        # None oznacza pracę przez cały opad; liczba oznacza czas pracy w sekundach.
        active_thrust = (
            thrust
            if thrust_duration is None or elapsed < thrust_duration
            else 0.0
        )
        acceleration_east = (drag_east + active_thrust * math.sin(heading)) / mass
        acceleration_north = (drag_north + active_thrust * math.cos(heading)) / mass
        acceleration_up = (drag_up / mass) - G

        # Pochylenie czaszy podąża z opóźnieniem za pozornym wiatrem.
        relative_horizontal_speed = math.hypot(relative_east, relative_north)
        relative_down_speed = max(-relative_up, 0.5)
        target_tilt_rad = math.atan2(relative_horizontal_speed, relative_down_speed)
        target_tilt_rad = min(target_tilt_rad, math.radians(75))
        relative_downrange = (
            relative_east * math.sin(heading)
            + relative_north * math.cos(heading)
        )
        tilt_direction = (
            -1.0 if relative_downrange > 0
            else 1.0 if relative_downrange < 0
            else 0.0
        )
        canopy_tilt_rad += (target_tilt_rad - canopy_tilt_rad) * min(
            time_step / canopy_lag_seconds, 1.0
        )
        tilt_deg = math.degrees(canopy_tilt_rad)
        maximum_tilt_deg = max(maximum_tilt_deg, tilt_deg)

        # Krok półjawny: najpierw aktualizujemy prędkość, potem położenie.
        velocity_east += acceleration_east * time_step
        velocity_north += acceleration_north * time_step
        velocity_up += acceleration_up * time_step
        next_east = east + velocity_east * time_step
        next_north = north + velocity_north * time_step
        next_altitude = altitude + velocity_up * time_step
        next_elapsed = elapsed + time_step

        # Gdy krok przechodzi przez ziemię, dokładniej wyznaczamy moment i miejsce lądowania.
        if next_altitude <= 0:
            fraction = altitude / (altitude - next_altitude)
            elapsed += time_step * fraction
            east += (next_east - east) * fraction
            north += (next_north - north) * fraction
            altitude = 0.0
            history["time"].append(elapsed)
            history["east"].append(east)
            history["north"].append(north)
            history["altitude"].append(altitude)
            history["speed_horizontal"].append(math.hypot(velocity_east, velocity_north))
            history["speed_vertical"].append(abs(velocity_up))
            history["tilt_deg"].append(tilt_deg)
            history["tilt_direction"].append(tilt_direction)
            break

        east, north, altitude, elapsed = next_east, next_north, next_altitude, next_elapsed
        history["time"].append(elapsed)
        history["east"].append(east)
        history["north"].append(north)
        history["altitude"].append(altitude)
        history["speed_horizontal"].append(math.hypot(velocity_east, velocity_north))
        history["speed_vertical"].append(abs(velocity_up))
        history["tilt_deg"].append(tilt_deg)
        history["tilt_direction"].append(tilt_direction)

    history["maximum_tilt_deg"] = maximum_tilt_deg
    history["landed"] = altitude <= 0
    return history


class CanSatSimApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Symulator Aerodynamiki CanSata")
        self.setGeometry(100, 100, 1100, 700)
        self.initUI()

    def initUI(self):
        self.setMinimumSize(1100, 760)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Ustawienia i wyniki są przewijane, żeby mieściły się także na mniejszym ekranie.
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(370)
        scroll_area.setWidget(controls_widget)
        main_layout.addWidget(scroll_area, 0)

        # Suwaki ustawiają parametry zrzutu; wartości obok od razu pokazują jednostki.
        controls_box = QGroupBox("Warunki zrzutu")
        controls_grid = QGridLayout(controls_box)
        self.sliders = {}
        self.value_labels = {}

        def add_slider(key, title, minimum, maximum, initial, format_value, row):
            title_label = QLabel(title)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(minimum, maximum)
            slider.setValue(initial)
            value_label = QLabel(format_value(initial))
            value_label.setMinimumWidth(68)
            controls_grid.addWidget(title_label, row, 0)
            controls_grid.addWidget(slider, row, 1)
            controls_grid.addWidget(value_label, row, 2)
            slider.valueChanged.connect(self.update_simulation)
            self.sliders[key] = slider
            self.value_labels[key] = (value_label, format_value)

        add_slider("mass", "Masa całkowita", 200, 800, 350,
                   lambda value: f"{value} g", 0)
        add_slider("area", "Powierzchnia czaszy", 50, 300, 110,
                   lambda value: f"{value / 1000:.3f} m²", 1)
        add_slider("height", "Wysokość otwarcia", 100, 3000, 1000,
                   lambda value: f"{value} m", 2)
        add_slider("thrust", "Łączny ciąg 2 śmigieł", 0, 100, 8,
                   lambda value: f"{value / 10:.1f} N", 3)
        add_slider("duration", "Czas pracy śmigieł", 0, 300, 50,
                   lambda value: f"{value / 10:.1f} s", 4)
        add_slider("wind_speed", "Prędkość wiatru", 0, 200, 50,
                   lambda value: f"{value / 10:.1f} m/s", 5)
        add_slider("wind_from", "Wiatr wieje z kierunku", 0, 359, 270,
                   lambda value: f"{value}°", 6)
        add_slider("heading", "Kierunek ciągu śmigieł", 0, 359, 90,
                   lambda value: f"{value}°", 7)

        # Ten tryb trzyma ustawiony ciąg aż do lądowania zamiast kończyć po czasie.
        self.full_flight_checkbox = QCheckBox("Utrzymuj ciąg do lądowania")
        self.full_flight_checkbox.setChecked(False)
        controls_grid.addWidget(self.full_flight_checkbox, 9, 0, 1, 3)
        self.full_flight_checkbox.toggled.connect(self.update_thrust_mode)

        # Współczynnik oporu zależy od kształtu i jest orientacyjny.
        controls_grid.addWidget(QLabel("Kształt spadochronu"), 8, 0)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(SHAPES.keys())
        self.shape_combo.currentIndexChanged.connect(self.update_simulation)
        controls_grid.addWidget(self.shape_combo, 8, 1, 1, 2)
        controls_layout.addWidget(controls_box)

        # Wyniki podsumowują czas lotu, miejsce lądowania i prędkość przy ziemi.
        results_box = QGroupBox("Przewidywane lądowanie")
        results_grid = QGridLayout(results_box)
        self.results = {}
        result_rows = [
            ("flight_time", "Czas lotu"),
            ("landing_offset", "Lądowanie względem startu"),
            ("vertical_speed", "Prędkość pionowa przy ziemi"),
            ("horizontal_speed", "Prędkość pozioma przy ziemi"),
            ("total_speed", "Prędkość wypadkowa przy ziemi"),
            ("tilt", "Maksymalne pochylenie czaszy"),
        ]
        for row, (key, title) in enumerate(result_rows):
            results_grid.addWidget(QLabel(title), row, 0)
            value_label = QLabel("-")
            value_label.setTextFormat(Qt.RichText)
            results_grid.addWidget(value_label, row, 1)
            self.results[key] = value_label
        controls_layout.addWidget(results_box)

        # Informujemy, które zjawiska pomija uproszczony model punktu masy.
        model_note = QLabel(
            "Model orientacyjny: oba śmigła mają wspólny ciąg. Nie uwzględnia "
            "niezależnego sterowania, turbulencji ani kołysania linek; nie służy "
            "do sterowania prawdziwym CanSatem."
        )
        model_note.setWordWrap(True)
        controls_layout.addWidget(model_note)

        # Wykresy pokazują tor opadania, wysokość w czasie i pochylenie czaszy.
        self.figure = plt.figure(figsize=(10, 7))
        grid = self.figure.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
        self.ax_path = self.figure.add_subplot(grid[:, 0])
        self.ax_altitude = self.figure.add_subplot(grid[0, 1])
        self.ax_canopy = self.figure.add_subplot(grid[1, 1])
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas, 1)

        self.update_simulation()

    def update_thrust_mode(self, active):
        """Włącza lub wyłącza suwak czasu pracy zależnie od wybranego trybu."""
        self.sliders["duration"].setEnabled(not active)
        label, formatter = self.value_labels["duration"]
        label.setText("do lądowania" if active else formatter(self.sliders["duration"].value()))
        self.update_simulation()

    def update_simulation(self):
        # Odczytujemy suwaki i zamieniamy ich wartości na jednostki fizyczne.
        parameters = {key: slider.value() for key, slider in self.sliders.items()}
        for key, value in parameters.items():
            label, formatter = self.value_labels[key]
            if key == "duration" and self.full_flight_checkbox.isChecked():
                label.setText("do lądowania")
            else:
                label.setText(formatter(value))

        mass = parameters["mass"] / 1000.0
        canopy_area = parameters["area"] / 1000.0
        release_height = float(parameters["height"])
        total_thrust = parameters["thrust"] / 10.0
        thrust_duration = (
            None if self.full_flight_checkbox.isChecked()
            else parameters["duration"] / 10.0
        )
        wind_speed = parameters["wind_speed"] / 10.0
        wind_from_deg = float(parameters["wind_from"])
        heading_deg = float(parameters["heading"])
        drag_coefficient = SHAPES[self.shape_combo.currentText()]

        # Uruchamiamy model i dostajemy położenie oraz prędkość w każdej chwili lotu.
        flight = simulate_descent(
            mass, canopy_area, drag_coefficient, release_height,
            total_thrust, thrust_duration, wind_speed, wind_from_deg, heading_deg,
        )

        # Z końcowych prędkości wyliczamy wyniki przyziemienia.
        landing_east = flight["east"][-1]
        landing_north = flight["north"][-1]
        landing_distance = math.hypot(landing_east, landing_north)
        vertical_impact = flight["speed_vertical"][-1]
        horizontal_impact = flight["speed_horizontal"][-1]
        total_impact = math.hypot(vertical_impact, horizontal_impact)
        maximum_tilt = flight["maximum_tilt_deg"]

        # Aktualizujemy podsumowanie wyników w panelu po lewej stronie.
        result_text = {
            "flight_time": f"<b>{flight['time'][-1]:.1f} s</b>",
            "landing_offset": (
                f"<b>{landing_distance:.1f} m</b><br>"
                f"E: {landing_east:+.1f} m, N: {landing_north:+.1f} m"
            ),
            "vertical_speed": f"<b>{vertical_impact:.2f} m/s</b>",
            "horizontal_speed": f"<b>{horizontal_impact:.2f} m/s</b>",
            "total_speed": f"<b>{total_impact:.2f} m/s</b>",
            "tilt": f"<b>{maximum_tilt:.1f}°</b>",
        }
        for key, text in result_text.items():
            self.results[key].setText(text)

        # Rysujemy tor w płaszczyźnie kierunku silników i wysokość w czasie.
        heading = math.radians(heading_deg)
        downrange = [
            east * math.sin(heading) + north * math.cos(heading)
            for east, north in zip(flight["east"], flight["north"])
        ]
        self.ax_path.clear()
        self.ax_path.plot(downrange, flight["altitude"], color="#087e8b", linewidth=2)
        self.ax_path.scatter([downrange[0]], [flight["altitude"][0]], color="#2a9d8f", label="Start")
        self.ax_path.scatter([downrange[-1]], [0], color="#e76f51", label="Lądowanie")
        self.ax_path.set_title("Tor lotu")
        self.ax_path.set_xlabel("Odległość w kierunku ciągu (m)")
        self.ax_path.set_ylabel("Wysokość nad ziemią (m)")
        self.ax_path.set_ylim(0, release_height * 1.05)
        path_limit = max(max(abs(value) for value in downrange), 5.0) * 1.15
        self.ax_path.set_xlim(-path_limit, path_limit)
        self.ax_path.grid(True, linestyle="--", alpha=0.4)
        self.ax_path.legend(loc="best", fontsize=8)

        self.ax_altitude.clear()
        self.ax_altitude.plot(flight["time"], flight["altitude"], color="#264653", linewidth=2)
        self.ax_altitude.set_title("Wysokość w czasie")
        self.ax_altitude.set_xlabel("Czas (s)")
        self.ax_altitude.set_ylabel("Wysokość (m)")
        self.ax_altitude.grid(True, linestyle="--", alpha=0.4)

        # Schemat pokazuje największe obliczone pochylenie czaszy za poruszającą się puszką.
        maximum_tilt_index = flight["tilt_deg"].index(max(flight["tilt_deg"]))
        tilt_direction = flight["tilt_direction"][maximum_tilt_index]
        self.draw_cansat_system(maximum_tilt, tilt_direction, total_thrust > 0)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def draw_cansat_system(self, angle_deg, tilt_direction, thrust_active):
        """Rysuje schemat puszki i czaszy przechylonej przeciwnie do ruchu."""
        self.ax_canopy.clear()
        self.ax_canopy.set_title(f"Pochylenie czaszy: {angle_deg:.1f}°")
        self.ax_canopy.set_xlim(-3.0, 3.0)
        self.ax_canopy.set_ylim(-1.0, 4.0)
        self.ax_canopy.set_aspect("equal", adjustable="box")
        self.ax_canopy.axis("off")

        # Puszka pozostaje pionowa; pozioma strzałka oznacza kierunek ciągu śmigieł.
        body_width, body_height = 0.45, 0.8
        self.ax_canopy.add_patch(patches.Rectangle(
            (-body_width / 2, 0), body_width, body_height,
            edgecolor="#263238", facecolor="#cfd8dc", linewidth=2,
        ))
        if thrust_active:
            self.ax_canopy.annotate(
                "ciąg śmigieł",
                xy=(1.8, 0.4), xytext=(0.35, 0.4),
                arrowprops={"arrowstyle": "->", "color": "#e76f51", "lw": 2},
                color="#b54432", va="bottom",
            )

        # Linki i czasza odchylają się przeciwnie do poziomego ruchu względem powietrza.
        angle_rad = math.radians(angle_deg)
        canopy_x = tilt_direction * 1.7 * math.sin(angle_rad)
        canopy_y = body_height + 1.7 * math.cos(angle_rad)
        left_attachment = (canopy_x - 1.0, canopy_y)
        right_attachment = (canopy_x + 1.0, canopy_y)
        self.ax_canopy.plot(
            [-body_width / 2, left_attachment[0]], [body_height, left_attachment[1]],
            color="#455a64", linewidth=1.3,
        )
        self.ax_canopy.plot(
            [body_width / 2, right_attachment[0]], [body_height, right_attachment[1]],
            color="#455a64", linewidth=1.3,
        )
        canopy_transform = transforms.Affine2D().rotate_deg_around(
            canopy_x, canopy_y, -tilt_direction * angle_deg
        ) + self.ax_canopy.transData
        self.ax_canopy.add_patch(patches.Arc(
            (canopy_x, canopy_y), 2.2, 1.0,
            theta1=0, theta2=180, linewidth=2.5,
            color="#e76f51", transform=canopy_transform,
        ))
        self.ax_canopy.text(0, -0.55, "Schemat, nie w skali", ha="center", fontsize=8)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    sim = CanSatSimApp()
    sim.show()
    sys.exit(app.exec_())