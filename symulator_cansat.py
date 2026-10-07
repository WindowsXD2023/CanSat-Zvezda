"""Zintegrowany, orientacyjny symulator CanSata dla kampanii w Błędowie."""

from __future__ import annotations

import math
import sys
import csv
from dataclasses import dataclass
from datetime import datetime

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

G = 9.81
R_AIR = 287.05
SHAPES = {
    "Półsferyczny": 1.30,
    "Stożkowy": 0.95,
    "Płaski": 0.85,
    "Krzyżowy": 0.75,
    "Własny": 1.30,
}
PRESSURE_LEVELS = tuple(range(975, 499, -25))
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass
class WeatherProfile:
    heights_m: np.ndarray
    wind_east_ms: np.ndarray
    wind_north_ms: np.ndarray
    temperature_c: np.ndarray
    pressure_hpa: np.ndarray
    timestamp: str
    elevation_m: float

    def at(self, height_m: float) -> tuple[float, float, float, float, float]:
        return (
            float(np.interp(height_m, self.heights_m, self.wind_east_ms)),
            float(np.interp(height_m, self.heights_m, self.wind_north_ms)),
            float(np.interp(height_m, self.heights_m, self.temperature_c)),
            float(np.interp(height_m, self.heights_m, self.pressure_hpa)),
            self.elevation_m,
        )

    def shifted(self, east_ms: float, north_ms: float) -> WeatherProfile:
        return WeatherProfile(
            self.heights_m,
            self.wind_east_ms + east_ms,
            self.wind_north_ms + north_ms,
            self.temperature_c,
            self.pressure_hpa,
            self.timestamp,
            self.elevation_m,
        )


def _hourly_series(hourly: dict[str, object], key: str, count: int) -> np.ndarray:
    values = hourly.get(key, [])
    if not isinstance(values, list):
        return np.full(count, np.nan)
    series = np.full(count, np.nan)
    for index, value in enumerate(values[:count]):
        if value is not None:
            series[index] = float(value)
    return series


def _interpolate_profile(
    source_heights: np.ndarray,
    source_values: np.ndarray,
    target_heights: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(source_heights) & np.isfinite(source_values)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Open-Meteo zwróciło za mało punktów profilu.")
    order = np.argsort(source_heights[valid])
    heights = source_heights[valid][order]
    values = source_values[valid][order]
    heights, unique_indices = np.unique(heights, return_index=True)
    values = values[unique_indices]
    if len(heights) < 2:
        raise ValueError("Profil pogody musi zawierać co najmniej dwie wysokości.")
    return np.interp(target_heights, heights, values)


def fetch_open_meteo_profiles(
    latitude: float,
    longitude: float,
    model: str | None,
) -> list[WeatherProfile]:
    variables: list[str] = [
        "temperature_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
    ]
    for level in PRESSURE_LEVELS:
        variables.extend(
            (
                f"temperature_{level}hPa",
                f"wind_speed_{level}hPa",
                f"wind_direction_{level}hPa",
                f"geopotential_height_{level}hPa",
            )
        )
    params: dict[str, str | float | int] = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "wind_speed_unit": "ms",
        "timezone": "Europe/Warsaw",
        "forecast_days": 2,
    }
    if model:
        params["models"] = model

    response = requests.get(WEATHER_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Open-Meteo zwróciło nieprawidłowy format odpowiedzi.")
    hourly = data.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ValueError("Odpowiedź Open-Meteo nie zawiera prognozy godzinowej.")

    if not hourly["time"]:
        raise ValueError("Open-Meteo zwróciło pustą prognozę godzinową.")
    elevation = float(data.get("elevation", 0.0))
    time_count = len(hourly["time"])
    source_heights = np.vstack(
        [
            np.full(time_count, 10.0),
            *[
                _hourly_series(
                    hourly, f"geopotential_height_{level}hPa", time_count
                ) - elevation
                for level in PRESSURE_LEVELS
            ],
        ]
    )
    target_heights = np.arange(0.0, 3001.0, 50.0)
    profiles: list[WeatherProfile] = []

    for time_index, timestamp in enumerate(hourly["time"]):
        heights = source_heights[:, time_index]
        speeds = np.asarray(
            [
                _hourly_series(hourly, "wind_speed_10m", time_count)[time_index],
                *[
                    _hourly_series(
                        hourly, f"wind_speed_{level}hPa", time_count
                    )[time_index]
                    for level in PRESSURE_LEVELS
                ],
            ]
        )
        directions = np.deg2rad(
            np.asarray(
                [
                    _hourly_series(
                        hourly, "wind_direction_10m", time_count
                    )[time_index],
                    *[
                        _hourly_series(
                            hourly, f"wind_direction_{level}hPa", time_count
                        )[time_index]
                        for level in PRESSURE_LEVELS
                    ],
                ]
            )
        )
        temperatures = np.asarray(
            [
                _hourly_series(hourly, "temperature_2m", time_count)[time_index],
                *[
                    _hourly_series(
                        hourly, f"temperature_{level}hPa", time_count
                    )[time_index]
                    for level in PRESSURE_LEVELS
                ],
            ]
        )
        pressures = np.asarray(
            [
                _hourly_series(hourly, "surface_pressure", time_count)[time_index],
                *[float(level) for level in PRESSURE_LEVELS],
            ]
        )
        wind_east = -speeds * np.sin(directions)
        wind_north = -speeds * np.cos(directions)

        valid_height = np.isfinite(heights)
        if np.count_nonzero(valid_height) < 2 or np.nanmax(heights) < 2500:
            continue
        try:
            profiles.append(
                WeatherProfile(
                    heights_m=target_heights,
                    wind_east_ms=_interpolate_profile(
                        heights, wind_east, target_heights
                    ),
                    wind_north_ms=_interpolate_profile(
                        heights, wind_north, target_heights
                    ),
                    temperature_c=_interpolate_profile(
                        heights, temperatures, target_heights
                    ),
                    pressure_hpa=_interpolate_profile(
                        heights, pressures, target_heights
                    ),
                    timestamp=timestamp,
                    elevation_m=elevation,
                )
            )
        except ValueError:
            continue

    if not profiles:
        raise ValueError(
            "Open-Meteo nie zwróciło pełnego profilu do 2500 m dla tej lokalizacji."
        )
    return profiles


def manual_weather_profile(
    wind_speed_ms: float,
    wind_from_deg: float,
    temperature_c: float,
    pressure_hpa: float,
) -> WeatherProfile:
    heights = np.arange(0.0, 3501.0, 50.0)
    direction = math.radians(wind_from_deg)
    wind_east = np.full(len(heights), -wind_speed_ms * math.sin(direction))
    wind_north = np.full(len(heights), -wind_speed_ms * math.cos(direction))
    temperature = temperature_c - 0.0065 * heights
    pressure = pressure_hpa * np.maximum(
        (temperature + 273.15) / (temperature_c + 273.15), 0.1
    ) ** 5.25588
    return WeatherProfile(
        heights,
        wind_east,
        wind_north,
        temperature,
        pressure,
        "Pogoda ręczna (stały wiatr)",
        0.0,
    )


def parachute_area_for_speed(
    mass_kg: float, drag_coefficient: float, descent_speed_ms: float, density_kg_m3: float
) -> float:
    if min(mass_kg, drag_coefficient, descent_speed_ms, density_kg_m3) <= 0:
        raise ValueError("Masa, Cd, prędkość i gęstość muszą być dodatnie.")
    return 2 * mass_kg * G / (
        density_kg_m3 * drag_coefficient * descent_speed_ms**2
    )


def terminal_descent_speed(
    mass_kg: float, area_m2: float, drag_coefficient: float, density_kg_m3: float
) -> float:
    if min(mass_kg, area_m2, drag_coefficient, density_kg_m3) <= 0:
        raise ValueError("Masa, powierzchnia, Cd i gęstość muszą być dodatnie.")
    return math.sqrt(
        2 * mass_kg * G / (density_kg_m3 * drag_coefficient * area_m2)
    )


def geographic_to_local(
    latitude: float, longitude: float, origin_latitude: float, origin_longitude: float
) -> tuple[float, float]:
    east = (
        longitude - origin_longitude
    ) * 111320 * max(math.cos(math.radians(origin_latitude)), 0.01)
    north = (latitude - origin_latitude) * 110540
    return east, north


def wind_direction_from_components(east_ms: float, north_ms: float) -> float | None:
    """Meteorological bearing the wind comes from, clockwise from north."""
    if math.hypot(east_ms, north_ms) < 1e-6:
        return None
    return math.degrees(math.atan2(-east_ms, -north_ms)) % 360


def simulate_descent(
    mass_kg: float,
    canopy_area_m2: float,
    drag_coefficient: float,
    release_height_m: float,
    thrust_n: float,
    thrust_duration_s: float | None,
    heading_deg: float,
    weather: WeatherProfile,
    time_step_s: float = 0.1,
    canopy_tilt_effect: bool = True,
) -> dict[str, list[float] | float | bool]:
    if mass_kg <= 0 or canopy_area_m2 <= 0 or drag_coefficient <= 0:
        raise ValueError("Masa, powierzchnia i współczynnik oporu muszą być dodatnie.")
    if release_height_m <= 0 or time_step_s <= 0 or thrust_n < 0:
        raise ValueError("Wysokość i krok czasu muszą być dodatnie, a ciąg nieujemny.")

    heading = math.radians(heading_deg)
    east = north = 0.0
    altitude = release_height_m
    velocity_east = velocity_north = velocity_up = 0.0
    elapsed = 0.0
    canopy_tilt_rad = 0.0
    maximum_tilt_deg = 0.0
    history: dict[str, list[float] | float | bool] = {
        "time": [0.0],
        "east": [0.0],
        "north": [0.0],
        "altitude": [altitude],
        "speed_horizontal": [0.0],
        "speed_vertical": [0.0],
        "tilt_deg": [0.0],
        "vertical_support_factor": [1.0],
    }

    while altitude > 0 and elapsed < 600:
        wind_east, wind_north, temperature_c, pressure_hpa, _ = weather.at(altitude)
        relative_east = velocity_east - wind_east
        relative_north = velocity_north - wind_north
        relative_up = velocity_up
        relative_speed = math.sqrt(
            relative_east**2 + relative_north**2 + relative_up**2
        )
        horizontal_air_speed = math.hypot(relative_east, relative_north)
        target_tilt = math.atan2(horizontal_air_speed, max(-relative_up, 0.5))
        canopy_tilt_rad += (min(target_tilt, math.radians(75)) - canopy_tilt_rad) * min(
            time_step_s / 1.2, 1.0
        )
        tilt_deg = math.degrees(canopy_tilt_rad)
        maximum_tilt_deg = max(maximum_tilt_deg, tilt_deg)
        vertical_support_factor = (
            math.cos(canopy_tilt_rad) if canopy_tilt_effect else 1.0
        )
        density = max(pressure_hpa, 1.0) * 100 / (
            R_AIR * max(temperature_c + 273.15, 150.0)
        )
        drag_factor = 0.5 * density * drag_coefficient * canopy_area_m2 * relative_speed
        drag_east = -drag_factor * relative_east
        drag_north = -drag_factor * relative_north
        drag_up = -drag_factor * relative_up * vertical_support_factor

        active_thrust = (
            thrust_n
            if thrust_duration_s is None or elapsed < thrust_duration_s
            else 0.0
        )
        acceleration_east = (
            drag_east + active_thrust * math.sin(heading)
        ) / mass_kg
        acceleration_north = (
            drag_north + active_thrust * math.cos(heading)
        ) / mass_kg
        acceleration_up = drag_up / mass_kg - G

        velocity_east += acceleration_east * time_step_s
        velocity_north += acceleration_north * time_step_s
        velocity_up += acceleration_up * time_step_s
        next_east = east + velocity_east * time_step_s
        next_north = north + velocity_north * time_step_s
        next_altitude = altitude + velocity_up * time_step_s
        next_elapsed = elapsed + time_step_s

        if next_altitude <= 0:
            fraction = altitude / (altitude - next_altitude)
            elapsed += time_step_s * fraction
            east += (next_east - east) * fraction
            north += (next_north - north) * fraction
            altitude = 0.0
        else:
            east, north, altitude, elapsed = (
                next_east,
                next_north,
                next_altitude,
                next_elapsed,
            )
        for key, value in (
            ("time", elapsed),
            ("east", east),
            ("north", north),
            ("altitude", altitude),
            ("speed_horizontal", math.hypot(velocity_east, velocity_north)),
            ("speed_vertical", abs(velocity_up)),
            ("tilt_deg", tilt_deg),
            ("vertical_support_factor", vertical_support_factor),
        ):
            history[key].append(value)  # type: ignore[union-attr]
        if altitude <= 0:
            break

    history["maximum_tilt_deg"] = maximum_tilt_deg
    history["landed"] = altitude <= 0
    return history


class WeatherFetchThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, latitude: float, longitude: float, model: str | None):
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.model = model

    def run(self) -> None:
        try:
            self.succeeded.emit(
                fetch_open_meteo_profiles(self.latitude, self.longitude, self.model)
            )
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            self.failed.emit(str(error))


def _spin(
    minimum: float,
    maximum: float,
    value: float,
    decimals: int = 2,
    step: float = 0.1,
) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    widget.setValue(value)
    return widget


class CanSatSimulator(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CanSat — symulator lotu i lądowania")
        self.resize(1500, 920)
        self.weather_profiles: list[WeatherProfile] = []
        self.weather_worker: WeatherFetchThread | None = None
        self.weather_location: tuple[float, float] | None = None
        self._pending_weather_refresh = False
        self._updating = False
        self.trajectory_colorbar = None
        self._build_ui()
        self.update_all()

    def _add_spin(
        self,
        form: QFormLayout,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        suffix: str,
        decimals: int = 2,
        step: float = 0.1,
    ) -> QDoubleSpinBox:
        widget = _spin(minimum, maximum, value, decimals, step)
        widget.setSuffix(suffix)
        form.addRow(label, widget)
        widget.valueChanged.connect(self.update_all)
        return widget

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        self.setCentralWidget(root)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(470)
        scroll.setWidget(controls)
        layout.addWidget(scroll)

        flight_box = QGroupBox("Parametry CanSata i zrzutu")
        flight_form = QFormLayout(flight_box)
        self.mass = self._add_spin(flight_form, "Masa całkowita", 0.05, 5, 0.35, " kg")
        self.release_height = self._add_spin(
            flight_form, "Wysokość odłączenia", 10, 2500, 2300, " m", 0
        )
        self.canopy_area = self._add_spin(
            flight_form,
            "Powierzchnia czaszy (nie objętość)",
            50,
            50000,
            1100,
            " cm²",
            0,
            50,
        )
        self.shape = QComboBox()
        self.shape.addItems(SHAPES)
        self.shape.setCurrentText("Półsferyczny")
        self.shape.currentTextChanged.connect(self._shape_changed)
        flight_form.addRow("Typ spadochronu", self.shape)
        self.drag_coefficient = self._add_spin(
            flight_form, "Współczynnik Cd", 0.2, 2.5, 1.3, "", 2, 0.05
        )
        self.thrust = self._add_spin(
            flight_form, "Łączny ciąg obu silników", 0, 25, 0, " N", 2, 0.1
        )
        self.thrust_duration = self._add_spin(
            flight_form, "Czas pracy silników", 0, 300, 5, " s", 1, 1
        )
        self.heading = self._add_spin(
            flight_form, "Azymut ciągu (0° = północ)", 0, 359, 90, "°", 0, 1
        )
        self.continuous_thrust = QCheckBox("Utrzymuj ciąg do lądowania")
        self.continuous_thrust.toggled.connect(self._thrust_mode_changed)
        flight_form.addRow(self.continuous_thrust)
        self.canopy_tilt_effect = QCheckBox(
            "Pochylenie czaszy osłabia podparcie pionowe"
        )
        self.canopy_tilt_effect.setChecked(True)
        self.canopy_tilt_effect.toggled.connect(self.update_all)
        flight_form.addRow(self.canopy_tilt_effect)
        self.integration_step = _spin(0.05, 0.5, 0.1, 2, 0.05)
        self.integration_step.setSuffix(" s")
        self.integration_step.valueChanged.connect(self.update_all)
        flight_form.addRow("Krok obliczeń", self.integration_step)
        controls_layout.addWidget(flight_box)

        location_box = QGroupBox("Lokalizacja i cel operatora")
        location_form = QFormLayout(location_box)
        self.latitude = self._add_spin(
            location_form, "Szerokość startu", -90, 90, 50.34, "°", 5, 0.001
        )
        self.longitude = self._add_spin(
            location_form, "Długość startu", -180, 180, 19.51, "°", 5, 0.001
        )
        self.latitude.editingFinished.connect(self._location_changed)
        self.longitude.editingFinished.connect(self._location_changed)
        self.target_latitude = self._add_spin(
            location_form, "Szerokość celu", -90, 90, 50.34, "°", 5, 0.001
        )
        self.target_longitude = self._add_spin(
            location_form, "Długość celu", -180, 180, 19.51, "°", 5, 0.001
        )
        controls_layout.addWidget(location_box)

        weather_box = QGroupBox("Pogoda — Open-Meteo domyślnie")
        weather_form = QFormLayout(weather_box)
        self.weather_mode = QComboBox()
        self.weather_mode.addItems(("Prognoza Open-Meteo", "Pogoda ręczna"))
        self.weather_mode.currentTextChanged.connect(self._weather_mode_changed)
        weather_form.addRow("Źródło danych", self.weather_mode)
        self.weather_model = QComboBox()
        self.weather_model.addItem("Automatyczny", "")
        self.weather_model.addItem("GFS Seamless", "gfs_seamless")
        weather_form.addRow("Model prognozy", self.weather_model)
        self.weather_model.currentIndexChanged.connect(self._weather_model_changed)
        self.forecast_time = QComboBox()
        self.forecast_time.currentIndexChanged.connect(self.update_all)
        weather_form.addRow("Godzina prognozy", self.forecast_time)
        self.fetch_button = QPushButton("Pobierz / odśwież prognozę")
        self.fetch_button.clicked.connect(self.fetch_weather)
        weather_form.addRow(self.fetch_button)
        self.manual_wind = self._add_spin(
            weather_form, "Wiatr (stały)", 0, 60, 5, " m/s", 1, 0.5
        )
        self.manual_direction = self._add_spin(
            weather_form, "Kierunek, skąd wieje", 0, 359, 270, "°", 0, 1
        )
        self.manual_temperature = self._add_spin(
            weather_form, "Temperatura przy ziemi", -60, 60, 15, " °C", 1, 0.5
        )
        self.manual_pressure = self._add_spin(
            weather_form, "Ciśnienie przy ziemi", 300, 1100, 1000, " hPa", 1, 1
        )
        self.weather_status = QLabel("Oczekiwanie na Open-Meteo…")
        self.weather_status.setWordWrap(True)
        weather_form.addRow("Status", self.weather_status)
        controls_layout.addWidget(weather_box)

        measurements_box = QGroupBox("Pomiar CanSata do porównania z prognozą")
        measurements_form = QFormLayout(measurements_box)
        self.measurement_height = self._add_spin(
            measurements_form, "Wysokość pomiaru", 0, 3000, 10, " m", 0, 1
        )
        self.measured_temperature = self._add_spin(
            measurements_form, "Zmierzona temperatura", -80, 80, 15, " °C", 1, 0.5
        )
        self.measured_pressure = self._add_spin(
            measurements_form, "Zmierz. ciśnienie", 100, 1200, 1000, " hPa", 1, 1
        )
        self.measured_wind_speed = self._add_spin(
            measurements_form, "Zmierz. prędkość wiatru", 0, 100, 5, " m/s", 1, 0.5
        )
        self.measured_wind_direction = self._add_spin(
            measurements_form, "Zmierz. kierunek (skąd)", 0, 359, 270, "°", 0, 1
        )
        controls_layout.addWidget(measurements_box)
        self.compare_result = QLabel("Porównanie pojawi się po wczytaniu pogody.")
        self.compare_result.setWordWrap(True)
        controls_layout.addWidget(self.compare_result)
        controls_layout.addStretch(1)
        layout.addWidget(self._build_tabs(), 1)
        self._weather_mode_changed(self.weather_mode.currentText())

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        trajectory_tab = QWidget()
        trajectory_layout = QVBoxLayout(trajectory_tab)
        self.summary = QLabel("Obliczanie trajektorii…")
        self.summary.setWordWrap(True)
        trajectory_layout.addWidget(self.summary)
        self.export_button = QPushButton("Eksportuj aktualną trajektorię do CSV")
        self.export_button.clicked.connect(self.export_trajectory)
        trajectory_layout.addWidget(self.export_button)
        self.flight_figure = plt.figure(figsize=(11, 7))
        grid = self.flight_figure.add_gridspec(
            3, 2, width_ratios=(1.35, 1), height_ratios=(1, 1, 0.85)
        )
        self.ax_3d = self.flight_figure.add_subplot(grid[:, 0], projection="3d")
        self.ax_altitude = self.flight_figure.add_subplot(grid[0, 1])
        self.ax_speeds = self.flight_figure.add_subplot(grid[1, 1])
        self.ax_tilt = self.flight_figure.add_subplot(grid[2, 1])
        self.flight_canvas = FigureCanvas(self.flight_figure)
        trajectory_layout.addWidget(self.flight_canvas, 1)
        self.calculation_notes = QLabel(
            "Wyniki korzystają z parametrów po lewej, wybranego profilu pogody "
            "i modelu oporu. Prognoza nie jest zastępowana wpisanymi pomiarami."
        )
        self.calculation_notes.setWordWrap(True)
        trajectory_layout.addWidget(self.calculation_notes)
        tabs.addTab(trajectory_tab, "Lot 3D")

        weather_tab = QWidget()
        weather_layout = QVBoxLayout(weather_tab)
        self.weather_figure = plt.figure(figsize=(11, 7))
        weather_grid = self.weather_figure.add_gridspec(2, 3)
        self.ax_wind = self.weather_figure.add_subplot(weather_grid[0, 0])
        self.ax_wind_direction = self.weather_figure.add_subplot(weather_grid[0, 1])
        self.ax_temperature = self.weather_figure.add_subplot(weather_grid[1, 0])
        self.ax_pressure = self.weather_figure.add_subplot(weather_grid[1, 1])
        self.ax_compass = self.weather_figure.add_subplot(
            weather_grid[:, 2], projection="polar"
        )
        self.weather_canvas = FigureCanvas(self.weather_figure)
        weather_layout.addWidget(self.weather_canvas, 1)
        self.profile_caption = QLabel(
            "Profil pionowy wiatru, temperatury i ciśnienia dla wybranej godziny."
        )
        self.profile_caption.setWordWrap(True)
        weather_layout.addWidget(self.profile_caption)
        tabs.addTab(weather_tab, "Profil pogody")

        design_tab = QWidget()
        design_layout = QVBoxLayout(design_tab)
        design_box = QGroupBox("Dobór spadochronu")
        design_form = QFormLayout(design_box)
        self.parachute_mode = QComboBox()
        self.parachute_mode.addItems(
            ("Powierzchnia dla zadanej prędkości", "Prędkość dla zadanej powierzchni")
        )
        self.parachute_mode.currentIndexChanged.connect(self.update_all)
        design_form.addRow("Co obliczyć?", self.parachute_mode)
        self.target_descent_speed = _spin(0.5, 30, 5, 2, 0.1)
        self.target_descent_speed.setSuffix(" m/s")
        self.target_descent_speed.valueChanged.connect(self.update_all)
        design_form.addRow("Docelowa prędkość opadania", self.target_descent_speed)
        self.max_tilt = _spin(1, 75, 30, 1, 1)
        self.max_tilt.setSuffix("°")
        self.max_tilt.valueChanged.connect(self.update_all)
        design_form.addRow("Granica akceptowanego pochylenia", self.max_tilt)
        self.design_result = QLabel()
        self.design_result.setWordWrap(True)
        design_form.addRow("Wynik doboru", self.design_result)
        self.apply_area_button = QPushButton("Ustaw wyliczoną powierzchnię w symulacji")
        self.apply_area_button.clicked.connect(self._apply_recommended_area)
        design_form.addRow(self.apply_area_button)
        design_layout.addWidget(design_box)

        self.shape_table = QTableWidget(4, 3)
        self.shape_table.setHorizontalHeaderLabels(
            ("Typ czaszy", "Powierzchnia dla celu", "Średnica równoważna")
        )
        self.shape_table.horizontalHeader().setStretchLastSection(True)
        design_layout.addWidget(QLabel("Porównanie kształtów dla zadanej prędkości"))
        design_layout.addWidget(self.shape_table)

        engine_box = QGroupBox("Silniki i korekta miejsca lądowania")
        engine_layout = QVBoxLayout(engine_box)
        self.engine_result = QLabel(
            "Model pochylenia jest przybliżeniem, nie wyznacza bezpiecznej granicy konstrukcji."
        )
        self.engine_result.setWordWrap(True)
        engine_layout.addWidget(self.engine_result)
        buttons = QHBoxLayout()
        self.thrust_limit_button = QPushButton("Oblicz limit ciągu")
        self.thrust_limit_button.clicked.connect(self.calculate_thrust_limit)
        buttons.addWidget(self.thrust_limit_button)
        self.correction_button = QPushButton("Szukaj poprawki do celu")
        self.correction_button.clicked.connect(self.calculate_correction)
        buttons.addWidget(self.correction_button)
        engine_layout.addLayout(buttons)
        design_layout.addWidget(engine_box)
        design_layout.addStretch(1)
        tabs.addTab(design_tab, "Projekt spadochronu i silników")

        scenarios_tab = QWidget()
        scenarios_layout = QVBoxLayout(scenarios_tab)
        self.scenario_button = QPushButton("Uruchom scenariusze i analizę niepewności")
        self.scenario_button.clicked.connect(self.run_scenarios)
        scenarios_layout.addWidget(self.scenario_button)
        self.ensemble_count = QSpinBox()
        self.ensemble_count.setRange(10, 100)
        self.ensemble_count.setValue(30)
        self.ensemble_count.valueChanged.connect(self.update_all)
        self.wind_uncertainty = _spin(0, 15, 1.5, 1, 0.5)
        self.wind_uncertainty.setSuffix(" m/s (1σ)")
        scenario_controls = QHBoxLayout()
        scenario_controls.addWidget(QLabel("Liczba prób Monte Carlo:"))
        scenario_controls.addWidget(self.ensemble_count)
        scenario_controls.addWidget(QLabel("Niepewność składowych wiatru:"))
        scenario_controls.addWidget(self.wind_uncertainty)
        scenarios_layout.addLayout(scenario_controls)
        self.scenario_table = QTableWidget(0, 5)
        self.scenario_table.setHorizontalHeaderLabels(
            (
                "Scenariusz",
                "Czas",
                "Odległość celu",
                "Prędkość pionowa",
                "Pochylenie",
            )
        )
        self.scenario_table.horizontalHeader().setStretchLastSection(True)
        scenarios_layout.addWidget(self.scenario_table)
        self.scenario_details = QLabel(
            "Uruchomienie pokaże dokładnie, które parametry różnią się między "
            "wariantami i jakie dane wejściowe zostały użyte."
        )
        self.scenario_details.setWordWrap(True)
        scenarios_layout.addWidget(self.scenario_details)
        self.scenario_figure = plt.figure(figsize=(8, 5))
        self.ax_scenarios = self.scenario_figure.add_subplot(111)
        self.scenario_canvas = FigureCanvas(self.scenario_figure)
        scenarios_layout.addWidget(self.scenario_canvas, 1)
        self.scenario_summary = QLabel(
            "Próby losowe są analizą wrażliwości, a nie gwarancją rozrzutu w locie."
        )
        self.scenario_summary.setWordWrap(True)
        scenarios_layout.addWidget(self.scenario_summary)
        tabs.addTab(scenarios_tab, "Scenariusze")

        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        help_text = QLabel(
            "<h2>Jak czytać symulację</h2>"
            "<p>Ustaw wysokość odłączenia od rakiety, współrzędne startu i punkt "
            "docelowy operatora, masę CanSata, parametry czaszy oraz pogodę. "
            "Program od razu przelicza lot; przy pierwszym uruchomieniu pobiera "
            "profil Open-Meteo dla wybranej lokalizacji.</p>"
            "<p><b>Jednostka czaszy:</b> wpisuje się pole powierzchni w cm² "
            "(np. 1100 cm² = 0,11 m²), a nie cm³. Średnica kołowa podawana w "
            "wyniku jest średnicą równoważną; dla innego kształtu liczy się "
            "rzeczywiste pole rzutu i zmierzony współczynnik Cd.</p>"
            "<p><b>Lot 3D:</b> osie poziome to długość i szerokość geograficzna "
            "WGS84, a oś pionowa to wysokość n.p.m. Zielony punkt oznacza "
            "odłączenie na zadanej wysokości, czerwony przewidywane lądowanie, "
            "niebieska gwiazda cel operatora. Trajektoria jest przeliczana z "
            "lokalnych składowych w metrach na stopnie geograficzne.</p>"
            "<p><b>Pogoda:</b> wybierz godzinę prognozy albo ręczne, stałe "
            "warunki. Ręczny wiatr to kierunek, <i>skąd</i> wieje. Ręczne "
            "ciśnienie i temperatura określają również gęstość powietrza.</p>"
            "<p><b>Projekt:</b> dobór czaszy opiera się na równowadze ciężaru "
            "i oporu w stanie ustalonym. Limit silników jest progiem "
            "pochylenia wybranym przez użytkownika. Poprawka szuka azymutu "
            "i czasu impulsu, ale nie jest sterownikiem w czasie rzeczywistym.</p>"
            "<p><b>Ograniczenia:</b> model punktu masy nie obejmuje turbulencji, "
            "kołysania i rozkładania czaszy, splątania linek ani dynamiki "
            "silników. Prognoza jest interpolowana z poziomów ciśnienia, nie "
            "jest pomiarem. Wynik nie zastępuje prób sprzętu ani oceny "
            "bezpieczeństwa kampanii.</p>"
            "<p><b>Pochylenie czaszy:</b> zaznaczony model orientacyjnie zmniejsza "
            "pionową składową siły oporu przez mnożnik cos(kąta pochylenia). "
            "Poziomy opór aerodynamiczny przeciwdziała ruchowi względem powietrza "
            "i hamuje składową poziomą. Większe pochylenie zmniejsza pionowe "
            "podparcie i może zwiększyć prędkość opadania. Mechanika czaszy, "
            "linek i wahadła nie jest rozwiązywana jako bryła.</p>"
            "<p><b>Scenariusze:</b> porównują wyłączone silniki, ustawienia "
            "bieżące, wiatr wzmocniony/osłabiony o 2 m/s w kierunku ciągu oraz "
            "model z wyłączonym sprzężeniem pochylenia. Monte Carlo wykonuje "
            "powtórzenia z losowym offsetem E/N wiatru. Szczegóły danych i "
            "różnic wariantów są pokazane pod tabelą. Przycisk CSV zapisuje "
            "punkty aktualnej trajektorii wraz z pozycją, prędkościami, "
            "pochyleniem i wiatrem.</p>"
        )
        help_text.setWordWrap(True)
        help_text.setTextFormat(Qt.RichText)
        help_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        help_layout.addWidget(help_text)
        help_layout.addStretch(1)
        tabs.addTab(help_tab, "Pomoc i założenia")
        return tabs

    def _shape_changed(self, name: str) -> None:
        self.drag_coefficient.setValue(SHAPES[name])
        self.update_all()

    def _thrust_mode_changed(self, checked: bool) -> None:
        self.thrust_duration.setEnabled(not checked)
        self.update_all()

    def _weather_model_changed(self) -> None:
        if self.weather_mode.currentText() == "Prognoza Open-Meteo":
            self.fetch_weather()

    def _location_changed(self) -> None:
        if self.weather_mode.currentText() == "Prognoza Open-Meteo":
            self.fetch_weather()
        else:
            self.update_all()

    def _weather_mode_changed(self, mode: str) -> None:
        is_forecast = mode == "Prognoza Open-Meteo"
        self.weather_model.setEnabled(is_forecast)
        self.forecast_time.setEnabled(is_forecast)
        self.fetch_button.setEnabled(is_forecast)
        for widget in (
            self.manual_wind,
            self.manual_direction,
            self.manual_temperature,
            self.manual_pressure,
        ):
            widget.setEnabled(not is_forecast)
        if not is_forecast:
            self.weather_status.setText(
                "Aktywna pogoda ręczna: stały wiatr i ręcznie ustawione "
                "temperatura oraz ciśnienie."
            )
        self.update_all()
        if (
            is_forecast
            and self.weather_location
            != (self.latitude.value(), self.longitude.value())
        ):
            self.fetch_weather()

    def fetch_weather(self) -> None:
        location = (self.latitude.value(), self.longitude.value())
        if self.weather_location != location:
            self.weather_profiles = []
            self.weather_location = None
            self.update_all()
        if self.weather_worker is not None and self.weather_worker.isRunning():
            self._pending_weather_refresh = True
            self.weather_status.setText(
                "Lokalizacja lub model uległy zmianie; odświeżę prognozę po "
                "zakończeniu bieżącego pobierania."
            )
            return
        self._pending_weather_refresh = False
        self.fetch_button.setEnabled(False)
        self.weather_status.setText("Pobieranie prognozy Open-Meteo…")
        self.weather_worker = WeatherFetchThread(
            self.latitude.value(),
            self.longitude.value(),
            self.weather_model.currentData(),
        )
        self.weather_worker.succeeded.connect(self._weather_loaded)
        self.weather_worker.failed.connect(self._weather_failed)
        self.weather_worker.finished.connect(self._weather_finished)
        self.weather_worker.start()

    def _weather_loaded(self, profiles: list[WeatherProfile]) -> None:
        if self.weather_worker is None:
            return
        location = (self.weather_worker.latitude, self.weather_worker.longitude)
        if location != (self.latitude.value(), self.longitude.value()):
            self._pending_weather_refresh = True
            return
        self.weather_profiles = profiles
        self.weather_location = location
        self.forecast_time.blockSignals(True)
        self.forecast_time.clear()
        for profile in profiles:
            timestamp = datetime.fromisoformat(profile.timestamp)
            self.forecast_time.addItem(timestamp.strftime("%d.%m %H:%M"))
        now = datetime.now()
        future_indices = [
            index
            for index, profile in enumerate(profiles)
            if datetime.fromisoformat(profile.timestamp) >= now
        ]
        if future_indices:
            self.forecast_time.setCurrentIndex(future_indices[0])
        self.forecast_time.blockSignals(False)
        self.weather_status.setText(
            f"Prognoza Open-Meteo: {len(profiles)} profili godzinowych, "
            f"elewacja {profiles[0].elevation_m:.0f} m n.p.m.; "
            f"lokalizacja {location[0]:.5f}° N, {location[1]:.5f}° E; "
            f"model {self.weather_model.currentText()}."
        )
        self.update_all()

    def _weather_failed(self, message: str) -> None:
        self.weather_profiles = []
        self.weather_location = None
        self.weather_status.setText(
            "Nie udało się pobrać prognozy; brak cichego zastępowania danymi "
            f"ręcznymi. Wybierz pogodę ręczną albo spróbuj ponownie. Szczegóły: {message}"
        )
        self.update_all()

    def _weather_finished(self) -> None:
        if self._pending_weather_refresh:
            self._pending_weather_refresh = False
            self.fetch_weather()
            return
        self.fetch_button.setEnabled(
            self.weather_mode.currentText() == "Prognoza Open-Meteo"
        )

    def _profile(self) -> WeatherProfile | None:
        if self.weather_mode.currentText() == "Pogoda ręczna":
            return manual_weather_profile(
                self.manual_wind.value(),
                self.manual_direction.value(),
                self.manual_temperature.value(),
                self.manual_pressure.value(),
            )
        index = self.forecast_time.currentIndex()
        if (
            self.weather_location == (self.latitude.value(), self.longitude.value())
            and 0 <= index < len(self.weather_profiles)
        ):
            return self.weather_profiles[index]
        return None

    def _flight(self, profile: WeatherProfile, thrust: float | None = None,
                duration: float | None = None, heading: float | None = None,
                time_step: float | None = None,
                tilt_effect: bool | None = None) -> dict[str, list[float] | float | bool]:
        return simulate_descent(
            self.mass.value(),
            self.canopy_area.value() / 10_000,
            self.drag_coefficient.value(),
            self.release_height.value(),
            self.thrust.value() if thrust is None else thrust,
            (
                None
                if self.continuous_thrust.isChecked()
                else self.thrust_duration.value()
            )
            if duration is None
            else duration,
            self.heading.value() if heading is None else heading,
            profile,
            self.integration_step.value() if time_step is None else time_step,
            self.canopy_tilt_effect.isChecked() if tilt_effect is None else tilt_effect,
        )

    @staticmethod
    def _value(flight: dict[str, list[float] | float | bool], name: str) -> list[float]:
        return flight[name]  # type: ignore[return-value]

    def update_all(self, *_args: object) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            profile = self._profile()
            if profile is None:
                self.summary.setText(
                    "Brak profilu pogody. Oczekuj na Open-Meteo albo przełącz się "
                    "na pogodę ręczną."
                )
                self.compare_result.setText("Brak prognozy do porównania z pomiarem.")
                self.ax_3d.clear()
                self.ax_altitude.clear()
                self.ax_speeds.clear()
                self.ax_tilt.clear()
                self.ax_wind.clear()
                self.ax_wind_direction.clear()
                self.ax_temperature.clear()
                self.ax_pressure.clear()
                self.ax_compass.clear()
                self.ax_compass.set_axis_off()
                self.ax_3d.text2D(
                    0.5,
                    0.5,
                    "Pobieram profil Open-Meteo…",
                    transform=self.ax_3d.transAxes,
                    ha="center",
                )
                self.ax_altitude.text(
                    0.5, 0.5, "Oczekiwanie na dane", transform=self.ax_altitude.transAxes,
                    ha="center", va="center"
                )
                self.ax_wind.text(
                    0.5, 0.5, "Oczekiwanie na dane", transform=self.ax_wind.transAxes,
                    ha="center", va="center"
                )
                self.ax_wind_direction.text(
                    0.5,
                    0.5,
                    "Oczekiwanie na dane",
                    transform=self.ax_wind_direction.transAxes,
                    ha="center",
                    va="center",
                )
                self.flight_canvas.draw_idle()
                self.weather_canvas.draw_idle()
                return
            flight = self._flight(profile)
            if not flight["landed"]:
                raise ValueError("Symulacja nie osiągnęła gruntu w limicie 600 s.")
            self.current_flight = flight
            self.current_profile = profile
            self._update_flight_plot(flight, profile)
            self._update_weather_plot(profile)
            self._update_measurement_comparison(profile)
            self._update_design(profile, flight)
        except ValueError as error:
            self.summary.setText(f"Nie można obliczyć trajektorii: {error}")
        finally:
            self._updating = False

    def _update_flight_plot(
        self,
        flight: dict[str, list[float] | float | bool],
        profile: WeatherProfile,
    ) -> None:
        east = self._value(flight, "east")
        north = self._value(flight, "north")
        altitude = self._value(flight, "altitude")
        times = self._value(flight, "time")
        tilt = self._value(flight, "tilt_deg")
        latitude = self.latitude.value()
        longitude = self.longitude.value()
        latitudes = [latitude + value / 110540 for value in north]
        longitude_scale = 111320 * max(math.cos(math.radians(latitude)), 0.01)
        longitudes = [longitude + value / longitude_scale for value in east]
        altitude_asl = [profile.elevation_m + value for value in altitude]
        target_east, target_north = geographic_to_local(
            self.target_latitude.value(),
            self.target_longitude.value(),
            latitude,
            longitude,
        )
        distance = math.hypot(east[-1] - target_east, north[-1] - target_north)
        maximum_tilt = float(flight["maximum_tilt_deg"])
        vertical_factor = self._value(flight, "vertical_support_factor")
        self.summary.setText(
            f"<b>Scenariusz:</b> odłączenie {self.release_height.value():.0f} m "
            f"AGL nad {latitude:.5f}° N, {longitude:.5f}° E; masa "
            f"{self.mass.value():.3f} kg; czasza {self.canopy_area.value():.0f} cm² "
            f"(Cd={self.drag_coefficient.value():.2f}); ciąg {self.thrust.value():.2f} N."
            f"<br><b>Czas lotu:</b> {times[-1]:.1f} s &nbsp; "
            f"<b>Prędkość pionowa przy ziemi:</b> "
            f"{self._value(flight, 'speed_vertical')[-1]:.2f} m/s &nbsp; "
            f"<b>Pozioma:</b> {self._value(flight, 'speed_horizontal')[-1]:.2f} m/s"
            f"<br><b>Lądowanie:</b> {latitudes[-1]:.6f}, {longitudes[-1]:.6f} "
            f"(WGS84, lokalne przybliżenie) &nbsp; "
            f"<b>Odległość od celu:</b> {distance:.1f} m &nbsp; "
            f"<b>Maks. pochylenie:</b> {maximum_tilt:.1f}° "
            f"(najmniejszy pionowy mnożnik oporu {min(vertical_factor):.2f})"
        )
        self.calculation_notes.setText(
            f"<b>Dane użyte:</b> {profile.timestamp}, źródło "
            f"{'Open-Meteo ' + self.weather_model.currentText() if self.weather_mode.currentText() == 'Prognoza Open-Meteo' else 'ręczna pogoda'}; "
            f"temperatura, ciśnienie oraz wektor wiatru E/N interpolowane z profilu "
            f"co 50 m (poziomy ciśnienia 975–500 hPa co 25 hPa). Start "
            f"{latitude:.5f}° N, {longitude:.5f}° E; wysokość "
            f"{self.release_height.value():.0f} m; masa {self.mass.value():.3f} kg; "
            f"czasza {self.canopy_area.value():.0f} cm², Cd={self.drag_coefficient.value():.2f}; "
            f"ciąg {self.thrust.value():.2f} N. <b>Obliczenie:</b> całkowanie sił "
            f"oporu i grawitacji co {self.integration_step.value():.2f} s aż do "
            "ziemi. Pomiar wpisany w panelu służy tylko porównaniu; nie zmienia "
            "prognozy ani symulacji."
        )
        self.ax_3d.clear()
        points = self.ax_3d.scatter(
            longitudes,
            latitudes,
            altitude_asl,
            c=altitude,
            cmap="viridis",
            s=7,
        )
        self.ax_3d.plot(
            longitudes, latitudes, altitude_asl, color="#087e8b", alpha=0.55
        )
        if self.trajectory_colorbar is None:
            self.trajectory_colorbar = self.flight_figure.colorbar(
                points,
                ax=self.ax_3d,
                shrink=0.55,
                pad=0.12,
                label="Wysokość AGL (m)",
            )
        else:
            self.trajectory_colorbar.update_normal(points)
        self.ax_3d.scatter(
            [longitude],
            [latitude],
            [profile.elevation_m + self.release_height.value()],
            color="green",
            marker="^",
            s=70,
            label="Odłączenie",
        )
        self.ax_3d.scatter(
            [longitudes[-1]], [latitudes[-1]], [profile.elevation_m],
            color="red", marker="o", label="Lądowanie"
        )
        self.ax_3d.scatter(
            [self.target_longitude.value()], [self.target_latitude.value()],
            [profile.elevation_m], color="blue", marker="*", s=80, label="Cel operatora"
        )
        self.ax_3d.set_title("Trajektoria 3D — współrzędne geograficzne")
        self.ax_3d.set_xlabel("Długość geogr. (°)")
        self.ax_3d.set_ylabel("Szerokość geogr. (°)")
        self.ax_3d.set_zlabel("Wysokość n.p.m. (m)")
        x_min, x_max = min(longitudes), max(longitudes)
        y_min, y_max = min(latitudes), max(latitudes)
        z_min = profile.elevation_m
        z_max = profile.elevation_m + self.release_height.value()
        x_span = max((x_max - x_min) * longitude_scale, 20.0)
        y_span = max((y_max - y_min) * 110540, 20.0)
        z_span = max(z_max - z_min, 20.0)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        self.ax_3d.set_xlim(
            x_mid - x_span / (2 * longitude_scale),
            x_mid + x_span / (2 * longitude_scale),
        )
        self.ax_3d.set_ylim(y_mid - y_span / 221080, y_mid + y_span / 221080)
        self.ax_3d.set_zlim(z_min, z_max)
        self.ax_3d.set_box_aspect(
            (
                x_span,
                y_span,
                z_span,
            )
        )
        self.ax_3d.view_init(elev=24, azim=-55)
        self.ax_3d.legend(fontsize=7)

        self.ax_altitude.clear()
        self.ax_altitude.plot(times, altitude, color="#264653")
        self.ax_altitude.set_title("Wysokość nad terenem")
        self.ax_altitude.set_xlabel("Czas (s)")
        self.ax_altitude.set_ylabel("Wysokość (m)")
        self.ax_altitude.grid(True, alpha=0.3)

        self.ax_speeds.clear()
        self.ax_speeds.plot(
            times,
            self._value(flight, "speed_horizontal"),
            color="#087e8b",
            label="Pozioma względem ziemi",
        )
        self.ax_speeds.plot(
            times,
            self._value(flight, "speed_vertical"),
            color="#e76f51",
            label="Pionowa",
        )
        self.ax_speeds.set_title("Prędkość CanSata")
        self.ax_speeds.set_xlabel("Czas (s)")
        self.ax_speeds.set_ylabel("m/s")
        self.ax_speeds.legend(fontsize=8)
        self.ax_speeds.grid(True, alpha=0.3)

        self.ax_tilt.clear()
        self.ax_tilt.plot(times, tilt, color="#e76f51")
        self.ax_tilt.axhline(self.max_tilt.value(), color="black", linestyle="--")
        self.ax_tilt.set_title("Szacowane pochylenie czaszy")
        self.ax_tilt.set_xlabel("Czas (s)")
        self.ax_tilt.set_ylabel("Kąt (°)")
        self.ax_tilt.grid(True, alpha=0.3)
        self.ax_tilt.text(
            0.01,
            0.98,
            (
                "Wpływ pochylenia: cos(kąt) × opór pionowy"
                if self.canopy_tilt_effect.isChecked()
                else "Wpływ pochylenia na opór pionowy wyłączony"
            ),
            transform=self.ax_tilt.transAxes,
            va="top",
            fontsize=8,
        )
        self.flight_figure.tight_layout()
        self.flight_canvas.draw_idle()

    def _update_weather_plot(self, profile: WeatherProfile) -> None:
        self.ax_wind.clear()
        self.ax_temperature.clear()
        self.ax_pressure.clear()
        wind_speed = np.hypot(profile.wind_east_ms, profile.wind_north_ms)
        self.ax_wind.plot(wind_speed, profile.heights_m, color="#087e8b")
        self.ax_wind.set_title("Wiatr")
        self.ax_wind.set_xlabel("m/s")
        self.ax_wind.set_ylabel("Wysokość n.p.t. (m)")
        wind_from = np.asarray(
            [
                wind_direction_from_components(east, north)
                if math.hypot(east, north) >= 1e-6
                else np.nan
                for east, north in zip(profile.wind_east_ms, profile.wind_north_ms)
            ],
            dtype=float,
        )
        self.ax_wind_direction.scatter(
            wind_from, profile.heights_m, c=wind_speed, cmap="viridis", s=12
        )
        self.ax_wind_direction.set_xlim(0, 360)
        self.ax_wind_direction.set_xticks(
            [0, 45, 90, 135, 180, 225, 270, 315, 360],
            ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"],
        )
        self.ax_wind_direction.set_title("Kierunek, SKĄD wieje")
        self.ax_wind_direction.set_xlabel("Azymut meteorologiczny")
        self.ax_wind_direction.set_ylabel("Wysokość n.p.t. (m)")
        sample_height = min(self.measurement_height.value(), float(profile.heights_m[-1]))
        self.ax_wind_direction.axhline(sample_height, color="#e76f51", linestyle="--")
        self.ax_temperature.plot(profile.temperature_c, profile.heights_m, color="#e76f51")
        self.ax_temperature.set_title("Temperatura")
        self.ax_temperature.set_xlabel("°C")
        self.ax_pressure.plot(profile.pressure_hpa, profile.heights_m, color="#264653")
        self.ax_pressure.set_title("Ciśnienie")
        self.ax_pressure.set_xlabel("hPa")
        for axis in (
            self.ax_wind,
            self.ax_wind_direction,
            self.ax_temperature,
            self.ax_pressure,
        ):
            axis.grid(True, alpha=0.3)
        self.ax_compass.clear()
        self.ax_compass.set_theta_zero_location("N")
        self.ax_compass.set_theta_direction(-1)
        self.ax_compass.set_ylim(0, 1)
        self.ax_compass.set_yticks([])
        self.ax_compass.set_title(
            f"Wiatr na {sample_height:.0f} m\nstrzałka wskazuje SKĄD",
            pad=18,
        )
        sample_east, sample_north, _, _, _ = profile.at(sample_height)
        compass_bearing = wind_direction_from_components(sample_east, sample_north)
        if compass_bearing is None:
            self.ax_compass.text(
                0.5, 0.5, "Cisza\nwiatrowa", transform=self.ax_compass.transAxes,
                ha="center", va="center"
            )
        else:
            theta_from = math.radians(compass_bearing)
            theta_to = (theta_from + math.pi) % (2 * math.pi)
            self.ax_compass.annotate(
                "",
                xy=(theta_from, 0.88),
                xytext=(theta_from, 0.12),
                arrowprops={"arrowstyle": "-|>", "color": "#e76f51", "lw": 3},
            )
            self.ax_compass.annotate(
                "",
                xy=(theta_to, 0.78),
                xytext=(theta_to, 0.15),
                arrowprops={"arrowstyle": "-|>", "color": "#087e8b", "lw": 2},
            )
            self.ax_compass.text(
                0.5,
                0.02,
                f"SKĄD {compass_bearing:.0f}° ({wind_speed[np.argmin(abs(profile.heights_m - sample_height))]:.1f} m/s)\n"
                f"KU {(compass_bearing + 180) % 360:.0f}°",
                transform=self.ax_compass.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
        self.weather_figure.tight_layout()
        self.weather_canvas.draw_idle()
        self.profile_caption.setText(
            f"Profil: {profile.timestamp}. Kierunek pokazuje, SKĄD wieje wiatr "
            "(0°=N, 90°=E); strzałka niebieska wskazuje kierunek, W KTÓRYM wieje. "
            "Open-Meteo pobiera temperature_2m, wind_speed_10m, "
            "wind_direction_10m, surface_pressure oraz dla poziomów 975–500 hPa "
            "co 25 hPa: temperaturę, prędkość/kierunek wiatru i wysokość "
            "geopotencjalną. Interpolacja co 50 m. Pogoda ręczna zakłada stały wiatr."
        )

    def _update_measurement_comparison(self, profile: WeatherProfile) -> None:
        height = self.measurement_height.value()
        wind_east, wind_north, temperature, pressure, _ = profile.at(height)
        forecast_speed = math.hypot(wind_east, wind_north)
        forecast_direction = wind_direction_from_components(wind_east, wind_north)
        forecast_direction = 0.0 if forecast_direction is None else forecast_direction
        wind_direction_error = (
            (forecast_direction - self.measured_wind_direction.value() + 180) % 360
        ) - 180
        self.compare_result.setText(
            f"Prognoza na {height:.0f} m, {profile.timestamp}: "
            f"T {temperature:.1f} °C (błąd pomiaru−prognoza "
            f"{self.measured_temperature.value() - temperature:+.1f} °C), "
            f"p {pressure:.1f} hPa "
            f"({self.measured_pressure.value() - pressure:+.1f} hPa), "
            f"wiatr {forecast_speed:.1f} m/s "
            f"({self.measured_wind_speed.value() - forecast_speed:+.1f} m/s), "
            f"kierunek SKĄD {forecast_direction:.0f}° "
            f"(różnica kierunku prognoza−pomiar {wind_direction_error:+.0f}°)."
        )

    def _update_design(
        self, profile: WeatherProfile, flight: dict[str, list[float] | float | bool]
    ) -> None:
        _, _, temperature, pressure, _ = profile.at(0)
        density = pressure * 100 / (R_AIR * (temperature + 273.15))
        cd = self.drag_coefficient.value()
        target_speed = self.target_descent_speed.value()
        area_for_target = parachute_area_for_speed(
            self.mass.value(), cd, target_speed, density
        )
        diameter_for_target = math.sqrt(4 * area_for_target / math.pi)
        speed_for_area = terminal_descent_speed(
            self.mass.value(), self.canopy_area.value() / 10_000, cd, density
        )
        if self.parachute_mode.currentIndex() == 0:
            self.design_result.setText(
                f"Dla Cd={cd:.2f}, m={self.mass.value():.3f} kg i "
                f"v={target_speed:.2f} m/s: A≈{area_for_target * 10000:.0f} cm² "
                f"({area_for_target:.3f} m²), średnica kołowej czaszy≈"
                f"{diameter_for_target * 100:.1f} cm. Prędkość dla aktualnej "
                f"powierzchni {self.canopy_area.value():.0f} cm²: "
                f"{speed_for_area:.2f} m/s."
            )
            self.apply_area_button.setEnabled(True)
        else:
            self.design_result.setText(
                f"Dla aktualnej czaszy A={self.canopy_area.value():.0f} cm² "
                f"i Cd={cd:.2f}: prędkość końcowa≈{speed_for_area:.2f} m/s. "
                f"Powierzchnia dla {target_speed:.2f} m/s wyniosłaby "
                f"{area_for_target * 10000:.0f} cm² (średnica kołowa "
                f"{diameter_for_target * 100:.1f} cm)."
            )
            self.apply_area_button.setEnabled(False)

        for row, (shape_name, shape_cd) in enumerate(
            list(SHAPES.items())[:4]
        ):
            area = parachute_area_for_speed(
                self.mass.value(), shape_cd, target_speed, density
            )
            diameter = math.sqrt(4 * area / math.pi)
            for column, text in enumerate(
                (
                    shape_name,
                    f"{area * 10000:.0f} cm² (Cd {shape_cd:.2f})",
                    f"{diameter * 100:.1f} cm",
                )
            ):
                self.shape_table.setItem(row, column, QTableWidgetItem(text))

        maximum_tilt = float(flight["maximum_tilt_deg"])
        self.engine_result.setText(
            f"Aktualny ciąg: {self.thrust.value():.2f} N, "
            f"czas: {'do lądowania' if self.continuous_thrust.isChecked() else f'{self.thrust_duration.value():.1f} s'}, "
            f"pochylenie modelu: {maximum_tilt:.1f}°. "
            f"Ustawiony limit: {self.max_tilt.value():.0f}°. "
            "Limit jest kryterium projektowym użytkownika, nie certyfikacją."
        )

    def _apply_recommended_area(self) -> None:
        profile = self._profile()
        if profile is None:
            return
        _, _, temperature, pressure, _ = profile.at(0)
        density = pressure * 100 / (R_AIR * (temperature + 273.15))
        area = parachute_area_for_speed(
            self.mass.value(),
            self.drag_coefficient.value(),
            self.target_descent_speed.value(),
            density,
        )
        self.canopy_area.setValue(area * 10_000)

    def calculate_thrust_limit(self) -> None:
        profile = self._profile()
        if profile is None:
            QMessageBox.warning(self, "Brak pogody", "Pobierz prognozę lub wybierz pogodę ręczną.")
            return
        self.thrust_limit_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            duration = (
                None
                if self.continuous_thrust.isChecked()
                else self.thrust_duration.value()
            )

            def safe(thrust: float) -> bool:
                flight = self._flight(profile, thrust=thrust, duration=duration, time_step=0.15)
                return bool(flight["landed"]) and float(flight["maximum_tilt_deg"]) <= self.max_tilt.value()

            if not safe(0):
                self.engine_result.setText(
                    "Już dla wyłączonych silników szacowane pochylenie przekracza "
                    "ustawiony limit. Zmień czaszę, warunki albo kryterium."
                )
                return
            candidate_thrusts = np.arange(0.5, 25.01, 0.5)
            unsafe_thrust = next(
                (float(candidate) for candidate in candidate_thrusts if not safe(float(candidate))),
                None,
            )
            if unsafe_thrust is None:
                self.engine_result.setText(
                    f"W skanowanym zakresie 0–25 N (co 0,5 N) nie znaleziono "
                    f"przekroczenia limitu {self.max_tilt.value():.0f}°. "
                    "Wynik nie określa limitu konstrukcyjnego."
                )
                return
            high = unsafe_thrust
            low = max(high - 0.5, 0.0)
            for _ in range(12):
                middle = (low + high) / 2
                if safe(middle):
                    low = middle
                else:
                    high = middle
            self.engine_result.setText(
                f"Granica z modelu: około {low:.2f} N łącznego ciągu "
                f"(limit pochylenia {self.max_tilt.value():.0f}°, "
                f"skan 0–25 N co 0,5 N, potem 12 podziałów przedziału). "
                "Używaj marginesu bezpieczeństwa; "
                "to nie zastępuje testów statycznych i prób lotnych."
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.thrust_limit_button.setEnabled(True)

    def calculate_correction(self) -> None:
        profile = self._profile()
        if profile is None:
            QMessageBox.warning(self, "Brak pogody", "Pobierz prognozę lub wybierz pogodę ręczną.")
            return
        if self.thrust.value() <= 0:
            self.engine_result.setText("Ustaw dodatni ciąg silników, aby szukać korekty.")
            return
        self.correction_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            launch_lat = self.latitude.value()
            launch_lon = self.longitude.value()
            east_per_degree = 111320 * max(
                math.cos(math.radians(launch_lat)), 0.01
            )
            target_east = (self.target_longitude.value() - launch_lon) * east_per_degree
            target_north = (self.target_latitude.value() - launch_lat) * 110540
            baseline = self._flight(profile, thrust=0, duration=0, time_step=0.15)
            max_duration = min(float(self._value(baseline, "time")[-1]), 180.0)
            durations = np.linspace(0.0, max_duration, 9)
            best: tuple[float, float, float, float, float] | None = None
            for heading in range(0, 360, 15):
                for duration in durations:
                    flight = self._flight(
                        profile,
                        thrust=self.thrust.value(),
                        duration=float(duration),
                        heading=float(heading),
                        time_step=0.15,
                    )
                    landed_east = float(self._value(flight, "east")[-1])
                    landed_north = float(self._value(flight, "north")[-1])
                    miss = math.hypot(landed_east - target_east, landed_north - target_north)
                    if best is None or miss < best[0]:
                        best = (miss, float(heading), float(duration), landed_east, landed_north)
            if best is None:
                raise ValueError("Nie znaleziono wariantu korekty.")
            self.engine_result.setText(
                f"Najlepszy przetestowany impuls: azymut {best[1]:.0f}°, "
                f"{best[2]:.1f} s przy {self.thrust.value():.2f} N; "
                f"przewidywane minięcie celu {best[0]:.1f} m. "
                "Szukano 24 azymutów co 15° oraz 9 czasów od 0 do czasu opadania "
                "(maks. 180 s), wybierając najmniejszą odległość punktu lądowania "
                "od celu. Użyto bieżącego profilu wiatru, masy, czaszy i ciągu; "
                "to propozycja modelowa, nie aktywne sterowanie w locie."
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.correction_button.setEnabled(True)

    def export_trajectory(self) -> None:
        flight = getattr(self, "current_flight", None)
        profile = getattr(self, "current_profile", None)
        if flight is None or profile is None:
            QMessageBox.warning(
                self,
                "Brak trajektorii",
                "Najpierw wybierz pogodę i przelicz poprawną trajektorię.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Zapisz trajektorię CanSata",
            "trajektoria_cansat.csv",
            "Pliki CSV (*.csv)",
        )
        if not path:
            return
        times = self._value(flight, "time")
        east = self._value(flight, "east")
        north = self._value(flight, "north")
        altitude = self._value(flight, "altitude")
        horizontal_speed = self._value(flight, "speed_horizontal")
        vertical_speed = self._value(flight, "speed_vertical")
        tilt = self._value(flight, "tilt_deg")
        latitude = self.latitude.value()
        longitude = self.longitude.value()
        longitude_scale = 111320 * max(math.cos(math.radians(latitude)), 0.01)
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    (
                        "czas_s",
                        "szerokosc_WGS84_deg",
                        "dlugosc_WGS84_deg",
                        "wysokosc_AGL_m",
                        "wysokosc_ASL_m",
                        "predkosc_pozioma_m_s",
                        "predkosc_pionowa_m_s",
                        "pochylenie_czaszy_deg",
                        "wiatr_speed_m_s",
                        "wiatr_skad_deg",
                        "profil_pogody",
                        "start_szerokosc_deg",
                        "start_dlugosc_deg",
                        "masa_kg",
                        "powierzchnia_czaszy_cm2",
                        "wspolczynnik_Cd",
                        "ciag_N",
                    )
                )
                for index, height in enumerate(altitude):
                    wind_east, wind_north, _, _, _ = profile.at(height)
                    wind_direction = wind_direction_from_components(
                        wind_east, wind_north
                    )
                    writer.writerow(
                        (
                            f"{times[index]:.3f}",
                            f"{latitude + north[index] / 110540:.7f}",
                            f"{longitude + east[index] / longitude_scale:.7f}",
                            f"{height:.2f}",
                            f"{profile.elevation_m + height:.2f}",
                            f"{horizontal_speed[index]:.3f}",
                            f"{vertical_speed[index]:.3f}",
                            f"{tilt[index]:.2f}",
                            f"{math.hypot(wind_east, wind_north):.3f}",
                            "" if wind_direction is None else f"{wind_direction:.1f}",
                            profile.timestamp,
                            f"{latitude:.7f}",
                            f"{longitude:.7f}",
                            f"{self.mass.value():.4f}",
                            f"{self.canopy_area.value():.1f}",
                            f"{self.drag_coefficient.value():.3f}",
                            f"{self.thrust.value():.3f}",
                        )
                    )
        except OSError as error:
            QMessageBox.critical(
                self,
                "Nie udało się zapisać CSV",
                f"Plik: {path}\nSzczegóły: {error}",
            )
            return
        QMessageBox.information(
            self, "Zapisano trajektorię", f"Zapisano {len(times)} punktów do:\n{path}"
        )

    def run_scenarios(self) -> None:
        profile = self._profile()
        if profile is None:
            QMessageBox.warning(self, "Brak pogody", "Pobierz prognozę lub wybierz pogodę ręczną.")
            return
        self.scenario_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            heading_rad = math.radians(self.heading.value())
            gust_east = 2 * math.sin(heading_rad)
            gust_north = 2 * math.cos(heading_rad)
            scenarios = [
                {
                    "name": "Bez silników",
                    "description": "ciąg=0 N; bieżąca pogoda i czasza",
                    "profile": profile,
                    "thrust": 0.0,
                    "duration": 0.0,
                    "tilt_effect": None,
                },
                {
                    "name": "Konfiguracja bieżąca",
                    "description": "wszystkie aktualne ustawienia panelu",
                    "profile": profile,
                    "thrust": None,
                    "duration": None,
                    "tilt_effect": None,
                },
                (
                    "Poryw zgodny +2 m/s",
                    profile.shifted(gust_east, gust_north),
                    None,
                    None,
                    None,
                    "dodaje +2 m/s do każdej wysokości, w kierunku silników",
                ),
                (
                    "Poryw przeciwny −2 m/s",
                    profile.shifted(-gust_east, -gust_north),
                    None,
                    None,
                    None,
                    "dodaje −2 m/s do każdej wysokości, przeciwnie do silników",
                ),
                (
                    "Bez sprzężenia pochylenia",
                    profile,
                    None,
                    None,
                    False,
                    "jak konfiguracja bieżąca, ale pionowy opór nie zależy od kąta",
                ),
            ]
            self.scenario_table.setRowCount(0)
            self.ax_scenarios.clear()
            landing_east: list[float] = []
            landing_north: list[float] = []
            target_east, target_north = geographic_to_local(
                self.target_latitude.value(),
                self.target_longitude.value(),
                self.latitude.value(),
                self.longitude.value(),
            )
            scenario_descriptions: list[str] = []
            for scenario in scenarios:
                if isinstance(scenario, dict):
                    name = scenario["name"]
                    description = scenario["description"]
                    scenario_profile = scenario["profile"]
                    thrust = scenario["thrust"]
                    duration = scenario["duration"]
                    tilt_effect = scenario["tilt_effect"]
                else:
                    (
                        name,
                        scenario_profile,
                        thrust,
                        duration,
                        tilt_effect,
                        description,
                    ) = scenario
                flight = self._flight(
                    scenario_profile,
                    thrust=thrust,
                    duration=duration,
                    time_step=0.15,
                    tilt_effect=tilt_effect,
                )
                east = float(self._value(flight, "east")[-1])
                north = float(self._value(flight, "north")[-1])
                landing_east.append(east)
                landing_north.append(north)
                miss_distance = math.hypot(
                    east - target_east, north - target_north
                )
                self._add_scenario_row(
                    name, flight, miss_distance, description
                )
                scenario_descriptions.append(f"{name}: {description}")

            rng = np.random.default_rng(2027)
            sigma = self.wind_uncertainty.value()
            count = self.ensemble_count.value()
            monte_carlo_east: list[float] = []
            monte_carlo_north: list[float] = []
            monte_carlo_errors: list[float] = []
            monte_carlo_times: list[float] = []
            monte_carlo_vertical_speeds: list[float] = []
            monte_carlo_tilts: list[float] = []
            for _ in range(count):
                perturbed = profile.shifted(
                    float(rng.normal(0, sigma)), float(rng.normal(0, sigma))
                )
                flight = self._flight(perturbed, time_step=0.15)
                east = float(self._value(flight, "east")[-1])
                north = float(self._value(flight, "north")[-1])
                miss = math.hypot(east - target_east, north - target_north)
                monte_carlo_east.append(east)
                monte_carlo_north.append(north)
                monte_carlo_errors.append(miss)
                monte_carlo_times.append(float(self._value(flight, "time")[-1]))
                monte_carlo_vertical_speeds.append(
                    float(self._value(flight, "speed_vertical")[-1])
                )
                monte_carlo_tilts.append(float(flight["maximum_tilt_deg"]))

            self.ax_scenarios.scatter(
                monte_carlo_east, monte_carlo_north, s=22, alpha=0.55,
                label=f"Monte Carlo ({count})"
            )
            self.ax_scenarios.scatter(
                landing_east, landing_north, marker="s", s=50,
                label="Scenariusze deterministyczne"
            )
            self.ax_scenarios.scatter(
                [target_east], [target_north], marker="*", color="red",
                s=140, label="Cel operatora"
            )
            self.ax_scenarios.set_title("Punkty lądowania (lokalne współrzędne)")
            self.ax_scenarios.set_xlabel("Wschód od startu (m)")
            self.ax_scenarios.set_ylabel("Północ od startu (m)")
            self.ax_scenarios.grid(True, alpha=0.3)
            self.ax_scenarios.axis("equal")
            self.ax_scenarios.legend()
            self.scenario_canvas.draw_idle()
            self.scenario_table.insertRow(self.scenario_table.rowCount())
            summary_values = (
                f"Monte Carlo: mediana ({count} prób)",
                f"{np.median(monte_carlo_times):.1f} s",
                f"{np.median(monte_carlo_errors):.1f} m",
                f"{np.median(monte_carlo_vertical_speeds):.2f} m/s",
                f"{np.median(monte_carlo_tilts):.1f}°",
            )
            for column, text in enumerate(summary_values):
                self.scenario_table.setItem(
                    self.scenario_table.rowCount() - 1,
                    column,
                    QTableWidgetItem(text),
                )
            self.scenario_summary.setText(
                f"Monte Carlo: {count} powtórzeń tej samej konfiguracji i wybranej "
                f"godziny pogody; w każdej próbie dodaję niezależny dla składowych "
                f"E i N, stały na całej wysokości offset N(0, {sigma:.1f}²) m/s. "
                f"Ziarno generatora=2027, więc wyniki są powtarzalne. "
                f"Odległość od celu: mediana {np.median(monte_carlo_errors):.1f} m, "
                f"95. percentyl {np.percentile(monte_carlo_errors, 95):.1f} m. "
                "Nie jest to prognoza prawdopodobieństwa: nie modeluje turbulencji, "
                "zmian pogody z czasem ani błędów czujników."
            )
            self.scenario_details.setText(
                f"<b>Dane wspólne:</b> {profile.timestamp}, "
                f"start {self.latitude.value():.5f}° N, "
                f"{self.longitude.value():.5f}° E; zrzut "
                f"{self.release_height.value():.0f} m; masa "
                f"{self.mass.value():.3f} kg; czasza "
                f"{self.canopy_area.value():.0f} cm², Cd "
                f"{self.drag_coefficient.value():.2f}; ciąg panelu "
                f"{self.thrust.value():.2f} N; krok obliczeń 0,15 s."
                f"<br><b>Co zmieniają warianty:</b> "
                f"{'; '.join(scenario_descriptions)}."
                "<br><b>Co liczy program:</b> dla każdego wariantu całkuje "
                "ruch aż do gruntu, oblicza prędkości, pochylenie, czas i "
                "odległość punktu lądowania od celu operatora."
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.scenario_button.setEnabled(True)

    def _add_scenario_row(
        self,
        name: str,
        flight: dict[str, list[float] | float | bool],
        miss_distance: float,
        description: str,
    ) -> None:
        row = self.scenario_table.rowCount()
        self.scenario_table.insertRow(row)
        values = (
            name,
            f"{self._value(flight, 'time')[-1]:.1f} s",
            f"{miss_distance:.1f} m",
            f"{self._value(flight, 'speed_vertical')[-1]:.2f} m/s",
            f"{float(flight['maximum_tilt_deg']):.1f}°",
        )
        for column, text in enumerate(values):
            item = QTableWidgetItem(text)
            if column == 0:
                item.setToolTip(description)
            self.scenario_table.setItem(row, column, item)


def main() -> int:
    app = QApplication(sys.argv)
    window = CanSatSimulator()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
