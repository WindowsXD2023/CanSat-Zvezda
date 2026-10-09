"""Integrated, approximate CanSat simulator for a campaign in Błędowo."""

from __future__ import annotations

import math
import sys
import csv
from collections.abc import Callable
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import LineCollection
import numpy as np
import requests
from PIL import Image
from PyQt5.QtGui import QCloseEvent
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
G = 9.81
EARTH_RADIUS_M = 6_371_000.0
R_AIR = 287.05
SHAPES = {
    "Hemispherical": 1.30,
    "Conical": 0.95,
    "Flat": 0.85,
    "Cross-shaped": 0.75,
    "Custom": 1.30,
}
PRESSURE_LEVELS = tuple(range(975, 499, -25))
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
SATELLITE_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


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
        raise ValueError("Open-Meteo returned too few profile points.")
    order = np.argsort(source_heights[valid])
    heights = source_heights[valid][order]
    values = source_values[valid][order]
    heights, unique_indices = np.unique(heights, return_index=True)
    values = values[unique_indices]
    if len(heights) < 2:
        raise ValueError("The weather profile must contain at least two altitudes.")
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
        raise ValueError("Open-Meteo returned an invalid response format.")
    hourly = data.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ValueError("The Open-Meteo response does not contain an hourly forecast.")

    if not hourly["time"]:
        raise ValueError("Open-Meteo returned an empty hourly forecast.")
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
            "Open-Meteo did not return a complete profile up to 2500 m for this location."
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
        "Manual weather (constant wind)",
        0.0,
    )


def parachute_area_for_speed(
    mass_kg: float, drag_coefficient: float, descent_speed_ms: float, density_kg_m3: float
) -> float:
    if min(mass_kg, drag_coefficient, descent_speed_ms, density_kg_m3) <= 0:
        raise ValueError("Mass, Cd, speed, and density must be positive.")
    return 2 * mass_kg * G / (
        density_kg_m3 * drag_coefficient * descent_speed_ms**2
    )


def terminal_descent_speed(
    mass_kg: float, area_m2: float, drag_coefficient: float, density_kg_m3: float
) -> float:
    if min(mass_kg, area_m2, drag_coefficient, density_kg_m3) <= 0:
        raise ValueError("Mass, area, Cd, and density must be positive.")
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


def local_to_geographic(
    east: float, north: float, origin_latitude: float, origin_longitude: float
) -> tuple[float, float]:
    latitude = origin_latitude + north / 110540
    longitude = origin_longitude + east / (
        111320 * max(math.cos(math.radians(origin_latitude)), 0.01)
    )
    return latitude, longitude


def wind_direction_from_components(east_ms: float, north_ms: float) -> float | None:
    """Meteorological bearing the wind comes from, clockwise from north."""
    if math.hypot(east_ms, north_ms) < 1e-6:
        return None
    return math.degrees(math.atan2(-east_ms, -north_ms)) % 360


def _plot_indices(point_count: int, max_points: int = 1200) -> np.ndarray:
    if point_count <= max_points:
        return np.arange(point_count)
    return np.linspace(0, point_count - 1, max_points, dtype=int)


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _sampled_reach_boundary(
    points: list[tuple[float, float]],
    baseline: tuple[float, float],
    angle_samples: int = 180,
) -> list[tuple[float, float]]:
    offsets = np.asarray(points, dtype=float) - np.asarray(baseline, dtype=float)
    radii = np.hypot(offsets[:, 0], offsets[:, 1])
    if not np.any(radii > 1e-6):
        return [baseline]

    angles = np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2 * math.pi)
    order = np.argsort(angles)
    sorted_angles = angles[order]
    sorted_radii = radii[order]
    unique_angles, starts = np.unique(sorted_angles, return_index=True)
    max_radii = np.maximum.reduceat(sorted_radii, starts)
    query_angles = np.linspace(0, 2 * math.pi, angle_samples, endpoint=False)
    extended_angles = np.concatenate(
        ([unique_angles[-1] - 2 * math.pi], unique_angles, [unique_angles[0] + 2 * math.pi])
    )
    extended_radii = np.concatenate(([max_radii[-1]], max_radii, [max_radii[0]]))
    boundary_radii = np.interp(query_angles, extended_angles, extended_radii)
    return [
        (
            baseline[0] + float(radius * math.cos(angle)),
            baseline[1] + float(radius * math.sin(angle)),
        )
        for angle, radius in zip(query_angles, boundary_radii)
    ]


def compute_reach_envelope(
    mass_kg: float,
    area_m2: float,
    drag_coefficient: float,
    release_height_m: float,
    max_thrust_n: float,
    heading_deg: float,
    weather: WeatherProfile,
    time_step_s: float = 0.75,
    heading_count: int = 24,
    burn_samples: int = 9,
    canopy_inflation_time_s: float = 0.0,
) -> dict[str, object]:
    """Sample asymmetric landing points for single continuous burns and coasting."""
    base = simulate_descent(
        mass_kg, area_m2, drag_coefficient, release_height_m, 0, 0,
        heading_deg, weather, time_step_s,
        canopy_inflation_time_s=canopy_inflation_time_s,
    )
    flight_seconds = float(base["time"][-1])
    burn_times = np.linspace(0, flight_seconds, burn_samples)
    baseline = (float(base["east"][-1]), float(base["north"][-1]))
    points = [baseline]
    for azimuth in np.linspace(0, 360, heading_count, endpoint=False):
        for burn_seconds in burn_times:
            flight = simulate_descent(
                mass_kg,
                area_m2,
                drag_coefficient,
                release_height_m,
                max_thrust_n,
                float(burn_seconds),
                float(azimuth),
                weather,
                time_step_s,
                canopy_inflation_time_s=canopy_inflation_time_s,
            )
            points.append((float(flight["east"][-1]), float(flight["north"][-1])))
    max_distance_from_baseline = max(
        math.hypot(east - baseline[0], north - baseline[1])
        for east, north in points
    )
    return {
        "points": points,
        "hull": _convex_hull(points),
        "boundary": _sampled_reach_boundary(points, baseline),
        "baseline": baseline,
        "max_distance_from_baseline_m": max_distance_from_baseline,
        "flight_seconds": flight_seconds,
        "samples": len(points),
        "max_thrust_n": max_thrust_n,
    }


def plan_navigation_to_target(
    mass_kg: float,
    area_m2: float,
    drag_coefficient: float,
    release_height_m: float,
    max_thrust_n: float,
    target_east_m: float,
    target_north_m: float,
    heading_deg: float,
    weather: WeatherProfile,
    time_step_s: float = 0.75,
    tolerance_m: float = 20.0,
    canopy_inflation_time_s: float = 0.0,
) -> dict[str, object]:
    """Search pulsed engine schedules with in-flight heading changes."""
    baseline_flight = simulate_descent(
        mass_kg, area_m2, drag_coefficient, release_height_m,
        0, 0, heading_deg, weather, time_step_s,
        canopy_inflation_time_s=canopy_inflation_time_s,
    )
    baseline_altitude = np.asarray(baseline_flight["altitude"], dtype=float)[::-1]
    baseline_east = np.asarray(baseline_flight["east"], dtype=float)[::-1]
    baseline_north = np.asarray(baseline_flight["north"], dtype=float)[::-1]
    baseline_landing = (
        float(baseline_flight["east"][-1]),
        float(baseline_flight["north"][-1]),
    )
    phase_count = 8
    cache: dict[
        tuple[int, int, int],
        tuple[float, float, dict[str, object]],
    ] = {}

    def evaluate(
        pulse_mask: int, horizon_index: int, phase_index: int
    ) -> tuple[float, float, dict[str, object]]:
        key = (pulse_mask, horizon_index, phase_index)
        if key not in cache:
            horizon_fraction = (0.65, 0.85)[horizon_index]
            schedule_horizon = float(baseline_flight["time"][-1]) * horizon_fraction
            slot_duration = schedule_horizon / phase_count
            phase_offset = slot_duration * phase_index / 3
            previous_heading = heading_deg % 360

            def guidance(
                elapsed: float,
                east: float,
                north: float,
                altitude: float,
                velocity_east: float,
                velocity_north: float,
                velocity_up: float,
            ) -> tuple[float, float]:
                nonlocal previous_heading
                del velocity_east, velocity_north, velocity_up
                baseline_position_east = float(
                    np.interp(altitude, baseline_altitude, baseline_east)
                )
                baseline_position_north = float(
                    np.interp(altitude, baseline_altitude, baseline_north)
                )
                drift_east = baseline_landing[0] - baseline_position_east
                drift_north = baseline_landing[1] - baseline_position_north
                target_offset_east = target_east_m - east - drift_east
                target_offset_north = target_north_m - north - drift_north
                desired_course = math.degrees(
                    math.atan2(target_offset_east, target_offset_north)
                ) % 360
                heading_delta = (
                    desired_course - previous_heading + 180
                ) % 360 - 180
                max_turn = 30.0 * time_step_s
                course = (
                    previous_heading
                    + max(-max_turn, min(max_turn, heading_delta))
                ) % 360
                previous_heading = course

                phase = int((elapsed + phase_offset) / slot_duration)
                engine_on = (
                    elapsed < schedule_horizon
                    and phase < phase_count
                    and bool(pulse_mask & (1 << phase))
                )
                return (max_thrust_n if engine_on else 0.0), course

            candidate = simulate_descent(
                mass_kg,
                area_m2,
                drag_coefficient,
                release_height_m,
                max_thrust_n,
                None,
                heading_deg,
                weather,
                time_step_s,
                control_function=guidance,
                canopy_inflation_time_s=canopy_inflation_time_s,
            )
            actual_burn_time = float(candidate["thrust_on_time_s"])
            flight_duration = float(candidate["time"][-1])
            if actual_burn_time > flight_duration * 0.85 + time_step_s * 0.01:
                miss = math.inf
            else:
                miss = math.hypot(
                    float(candidate["east"][-1]) - target_east_m,
                    float(candidate["north"][-1]) - target_north_m,
                )
            cache[key] = miss, actual_burn_time, candidate
        return cache[key]

    best_key = (0, 0, 0)
    best_score = (math.inf, math.inf)
    coarse_candidates: list[
        tuple[tuple[float, float], tuple[int, int, int]]
    ] = []
    for horizon_index in range(2):
        for pulse_mask in range(1 << phase_count):
            miss, burn_time, _ = evaluate(pulse_mask, horizon_index, 0)
            score = (miss, burn_time)
            key = (pulse_mask, horizon_index, 0)
            coarse_candidates.append((score, key))
            if score < best_score:
                best_key = key
                best_score = score

    for _, (pulse_mask, horizon_index, _) in sorted(coarse_candidates)[:12]:
        for phase_index in (1, 2):
            miss, burn_time, _ = evaluate(
                pulse_mask, horizon_index, phase_index
            )
            score = (miss, burn_time)
            if score < best_score:
                best_key = (pulse_mask, horizon_index, phase_index)
                best_score = score

    miss_distance, _, flight = evaluate(*best_key)
    heading_values = [
        float(value)
        for value, thrust in zip(flight["thrust_heading_deg"], flight["thrust_n"])
        if thrust > 1e-6
    ]
    initial_heading = heading_values[0] if heading_values else heading_deg % 360
    times = flight["time"]
    powered_indices = [
        index
        for index, thrust in enumerate(flight["thrust_n"])
        if thrust > 1e-6
    ]
    cutoff_index = powered_indices[-1] if powered_indices else 0
    previous_on = False
    active_transitions = 0
    engine_pulses = 0
    for thrust in flight["thrust_n"]:
        current_on = thrust > 1e-6
        if current_on != previous_on:
            active_transitions += 1
        if current_on and not previous_on:
            engine_pulses += 1
        previous_on = current_on
    return {
        "heading_deg": initial_heading,
        "heading_changes_during_flight": len(
            {round(value, 1) for value in heading_values}
        ) > 1,
        "thrust_on_time_s": float(flight["thrust_on_time_s"]),
        "miss_distance_m": miss_distance,
        "within_tolerance": miss_distance <= tolerance_m,
        "flight": flight,
        "control_samples": len(cache),
        "switch_count": active_transitions,
        "engine_pulses": engine_pulses,
        "cutoff_index": cutoff_index,
        "cutoff_east_m": float(flight["east"][cutoff_index]),
        "cutoff_north_m": float(flight["north"][cutoff_index]),
        "cutoff_altitude_m": float(flight["altitude"][cutoff_index]),
        "cutoff_time_s": float(times[cutoff_index]),
        "burn_duration_s": float(flight["thrust_on_time_s"]),
        "tolerance_m": tolerance_m,
        "max_thrust_n": max_thrust_n,
    }


def _tile_xy(latitude: float, longitude: float, zoom: int) -> tuple[int, int]:
    latitude = max(min(latitude, 85.05112878), -85.05112878)
    scale = 2**zoom
    x = int((longitude + 180) / 360 * scale)
    latitude_rad = math.radians(latitude)
    y = int(
        (1 - math.asinh(math.tan(latitude_rad)) / math.pi) / 2 * scale
    )
    return x, y


def fetch_satellite_mosaic(
    latitude_bounds: tuple[float, float],
    longitude_bounds: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float], int]:
    """Download a bounded, low-resolution Esri imagery mosaic for the 3D ground plane."""
    lat_min, lat_max = sorted(latitude_bounds)
    lon_min, lon_max = sorted(longitude_bounds)
    zoom = 17
    while zoom > 10:
        x_min, y_max = _tile_xy(lat_min, lon_min, zoom)
        x_max, y_min = _tile_xy(lat_max, lon_max, zoom)
        if (x_max - x_min + 1) * (y_max - y_min + 1) <= 25:
            break
        zoom -= 1
    x_min, y_max = _tile_xy(lat_min, lon_min, zoom)
    x_max, y_min = _tile_xy(lat_max, lon_max, zoom)
    mosaic = Image.new("RGB", ((x_max - x_min + 1) * 256, (y_max - y_min + 1) * 256))
    for tile_x in range(x_min, x_max + 1):
        for tile_y in range(y_min, y_max + 1):
            response = requests.get(
                SATELLITE_TILE_URL.format(z=zoom, y=tile_y, x=tile_x),
                timeout=8,
            )
            response.raise_for_status()
            tile = Image.open(BytesIO(response.content)).convert("RGB")
            mosaic.paste(
                tile,
                ((tile_x - x_min) * 256, (tile_y - y_min) * 256),
            )

    scale = 2**zoom

    def tile_longitude(x: int) -> float:
        return x / scale * 360 - 180

    def tile_latitude(y: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / scale))))

    extent = (
        tile_longitude(x_min),
        tile_longitude(x_max + 1),
        tile_latitude(y_max + 1),
        tile_latitude(y_min),
    )
    mosaic.thumbnail((768, 768), Image.Resampling.LANCZOS)
    return np.asarray(mosaic), extent, zoom


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
    control_function: Callable[
        [float, float, float, float, float, float, float],
        tuple[float, float],
    ] | None = None,
    canopy_inflation_time_s: float = 0.0,
) -> dict[str, list[float] | float | bool]:
    if mass_kg <= 0 or canopy_area_m2 <= 0 or drag_coefficient <= 0:
        raise ValueError("Mass, area, and drag coefficient must be positive.")
    if release_height_m <= 0 or time_step_s <= 0 or thrust_n < 0:
        raise ValueError("Altitude and time step must be positive, and thrust must be non-negative.")
    if thrust_duration_s is not None and thrust_duration_s < 0:
        raise ValueError("Motor run time cannot be negative.")
    if canopy_inflation_time_s < 0:
        raise ValueError("Canopy inflation time cannot be negative.")

    state = (0.0, 0.0, release_height_m, 0.0, 0.0, 0.0, 0.0)
    elapsed = 0.0
    maximum_tilt_deg = 0.0
    history: dict[str, list[float] | float | bool] = {
        "time": [0.0],
        "east": [0.0],
        "north": [0.0],
        "altitude": [release_height_m],
        "speed_horizontal": [0.0],
        "speed_vertical": [0.0],
        "tilt_deg": [0.0],
        "vertical_support_factor": [1.0],
        "canopy_inflation": [0.0 if canopy_inflation_time_s > 0 else 1.0],
        "thrust_n": [0.0],
        "thrust_heading_deg": [heading_deg],
        "dynamic_pressure_pa": [0.0],
        "specific_force_g": [0.0],
    }
    thrust_on_time_s = 0.0
    maximum_dynamic_pressure_pa = 0.0
    maximum_specific_force_g = 0.0
    held_control: tuple[float, float] | None = None
    control_interval_end = 0.0

    def inflation_fraction(time_s: float) -> float:
        if canopy_inflation_time_s <= 0:
            return 1.0
        return 1.0 - math.exp(-3.0 * max(time_s, 0.0) / canopy_inflation_time_s)

    def derivative(
        time_s: float,
        current: tuple[float, ...],
        applied_thrust_n: float,
        applied_heading_deg: float,
    ) -> tuple[float, ...]:
        _, _, height, velocity_east, velocity_north, velocity_up, tilt = current
        wind_east, wind_north, temperature_c, pressure_hpa, _ = weather.at(
            max(height, 0.0)
        )
        relative_east = velocity_east - wind_east
        relative_north = velocity_north - wind_north
        relative_up = velocity_up
        relative_speed = math.sqrt(
            relative_east**2 + relative_north**2 + relative_up**2
        )
        horizontal_air_speed = math.hypot(relative_east, relative_north)
        target_tilt = min(
            math.atan2(horizontal_air_speed, max(-relative_up, 0.5)),
            math.radians(75),
        )
        tilt_rate = (target_tilt - tilt) / 1.2
        density = max(pressure_hpa, 1.0) * 100 / (
            R_AIR * max(temperature_c + 273.15, 150.0)
        )
        drag_scale = (
            density
            * drag_coefficient
            * canopy_area_m2
            * inflation_fraction(time_s)
            * relative_speed
        )
        drag_east = -0.5 * drag_scale * relative_east
        drag_north = -0.5 * drag_scale * relative_north
        drag_up = (
            -0.5 * drag_scale * relative_up
            if canopy_tilt_effect
            else -0.5
            * density
            * drag_coefficient
            * canopy_area_m2
            * inflation_fraction(time_s)
            * abs(relative_up)
            * relative_up
        )
        heading_rad = math.radians(applied_heading_deg)
        acceleration_east = (
            drag_east + applied_thrust_n * math.sin(heading_rad)
        ) / mass_kg
        acceleration_north = (
            drag_north + applied_thrust_n * math.cos(heading_rad)
        ) / mass_kg
        local_gravity = G * (
            EARTH_RADIUS_M
            / (EARTH_RADIUS_M + weather.elevation_m + max(height, 0.0))
        ) ** 2
        acceleration_up = drag_up / mass_kg - local_gravity
        return (
            velocity_east,
            velocity_north,
            velocity_up,
            acceleration_east,
            acceleration_north,
            acceleration_up,
            tilt_rate,
        )

    def rk4_step(
        current: tuple[float, ...],
        time_s: float,
        step_s: float,
        applied_thrust_n: float,
        applied_heading_deg: float,
        first_derivative: tuple[float, ...] | None = None,
    ) -> tuple[float, ...]:
        k1 = first_derivative or derivative(
            time_s, current, applied_thrust_n, applied_heading_deg
        )
        k2_state = tuple(
            value + 0.5 * step_s * rate for value, rate in zip(current, k1)
        )
        k2 = derivative(
            time_s + 0.5 * step_s, k2_state, applied_thrust_n, applied_heading_deg
        )
        k3_state = tuple(
            value + 0.5 * step_s * rate for value, rate in zip(current, k2)
        )
        k3 = derivative(
            time_s + 0.5 * step_s, k3_state, applied_thrust_n, applied_heading_deg
        )
        k4_state = tuple(
            value + step_s * rate for value, rate in zip(current, k3)
        )
        k4 = derivative(
            time_s + step_s, k4_state, applied_thrust_n, applied_heading_deg
        )
        return tuple(
            value
            + step_s
            * (rate1 + 2 * rate2 + 2 * rate3 + rate4)
            / 6
            for value, rate1, rate2, rate3, rate4 in zip(
                current, k1, k2, k3, k4
            )
        )

    while state[2] > 0 and elapsed < 600:
        step_s = min(time_step_s, 600.0 - elapsed)
        if (
            control_function is None
            and thrust_duration_s is not None
            and elapsed < thrust_duration_s < elapsed + step_s
        ):
            step_s = thrust_duration_s - elapsed
        if step_s <= 1e-9:
            step_s = min(time_step_s, 600.0 - elapsed)

        applied_thrust_n, applied_heading_deg = thrust_n, heading_deg
        if control_function is not None:
            if held_control is None or elapsed >= control_interval_end - 1e-9:
                control_thrust_n, control_heading_deg = control_function(
                    elapsed,
                    state[0],
                    state[1],
                    state[2],
                    state[3],
                    state[4],
                    state[5],
                )
                held_control = (
                    min(max(control_thrust_n, 0.0), thrust_n),
                    control_heading_deg,
                )
                control_interval_end = elapsed + time_step_s
            applied_thrust_n, applied_heading_deg = held_control
            step_s = min(step_s, control_interval_end - elapsed)
        elif thrust_duration_s is not None and elapsed >= thrust_duration_s:
            applied_thrust_n = 0.0

        initial_derivative = derivative(
            elapsed, state, applied_thrust_n, applied_heading_deg
        )
        wind_east, wind_north, temperature_c, pressure_hpa, _ = weather.at(
            max(state[2], 0.0)
        )
        relative_speed = math.hypot(
            state[3] - wind_east,
            state[4] - wind_north,
            state[5],
        )
        density = max(pressure_hpa, 1.0) * 100 / (
            R_AIR * max(temperature_c + 273.15, 150.0)
        )
        drag_rate = (
            density
            * drag_coefficient
            * canopy_area_m2
            * inflation_fraction(elapsed)
            * relative_speed
            / mass_kg
        )
        acceleration = math.hypot(*initial_derivative[3:6])
        if drag_rate > 0:
            step_s = min(step_s, 0.8 / drag_rate)
        if acceleration > 0:
            step_s = min(step_s, 5.0 / acceleration)

        next_state = rk4_step(
            state,
            elapsed,
            step_s,
            applied_thrust_n,
            applied_heading_deg,
            initial_derivative,
        )
        actual_step_s = step_s
        if next_state[2] <= 0:
            low_s, high_s = 0.0, step_s
            for _ in range(14):
                middle_s = (low_s + high_s) / 2
                middle_state = rk4_step(
                    state,
                    elapsed,
                    middle_s,
                    applied_thrust_n,
                    applied_heading_deg,
                )
                if middle_state[2] > 0:
                    low_s = middle_s
                else:
                    high_s = middle_s
            actual_step_s = high_s
            next_state = rk4_step(
                state,
                elapsed,
                actual_step_s,
                applied_thrust_n,
                applied_heading_deg,
            )
            next_state = (*next_state[:2], 0.0, *next_state[3:])
        elapsed += actual_step_s
        state = next_state

        east, north, altitude, velocity_east, velocity_north, velocity_up, tilt = state
        tilt_deg = math.degrees(tilt)
        maximum_tilt_deg = max(maximum_tilt_deg, tilt_deg)
        wind_east, wind_north, temperature_c, pressure_hpa, _ = weather.at(
            max(altitude, 0.0)
        )
        relative_east = velocity_east - wind_east
        relative_north = velocity_north - wind_north
        relative_up = velocity_up
        relative_speed = math.sqrt(
            relative_east**2 + relative_north**2 + relative_up**2
        )
        vertical_support_factor = (
            abs(relative_up) / max(relative_speed, 1e-9)
            if canopy_tilt_effect
            else 1.0
        )
        density = max(pressure_hpa, 1.0) * 100 / (
            R_AIR * max(temperature_c + 273.15, 150.0)
        )
        dynamic_pressure = 0.5 * density * relative_speed**2
        drag_scale = (
            density
            * drag_coefficient
            * canopy_area_m2
            * inflation_fraction(elapsed)
            * relative_speed
        )
        drag_east = -0.5 * drag_scale * relative_east
        drag_north = -0.5 * drag_scale * relative_north
        drag_up = (
            -0.5 * drag_scale * relative_up
            if canopy_tilt_effect
            else -0.5
            * density
            * drag_coefficient
            * canopy_area_m2
            * inflation_fraction(elapsed)
            * abs(relative_up)
            * relative_up
        )
        specific_force_g = math.sqrt(
            (
                drag_east / mass_kg
                + applied_thrust_n
                * math.sin(math.radians(applied_heading_deg))
                / mass_kg
            ) ** 2
            + (
                drag_north / mass_kg
                + applied_thrust_n
                * math.cos(math.radians(applied_heading_deg))
                / mass_kg
            ) ** 2
            + (drag_up / mass_kg) ** 2
        ) / G
        maximum_dynamic_pressure_pa = max(
            maximum_dynamic_pressure_pa, dynamic_pressure
        )
        maximum_specific_force_g = max(maximum_specific_force_g, specific_force_g)
        active_thrust = applied_thrust_n
        if (
            control_function is None
            and thrust_duration_s is not None
            and elapsed - actual_step_s < thrust_duration_s < elapsed
        ):
            active_thrust *= max(
                0.0,
                min(1.0, (thrust_duration_s - (elapsed - actual_step_s)) / actual_step_s),
            )
        thrust_active_duration = (
            actual_step_s if active_thrust > 1e-6 else 0.0
        )
        if (
            control_function is None
            and thrust_duration_s is not None
            and elapsed - actual_step_s < thrust_duration_s < elapsed
        ):
            thrust_active_duration = max(
                0.0, thrust_duration_s - (elapsed - actual_step_s)
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
            ("canopy_inflation", inflation_fraction(elapsed)),
            ("thrust_n", active_thrust),
            ("thrust_heading_deg", applied_heading_deg),
            ("dynamic_pressure_pa", dynamic_pressure),
            ("specific_force_g", specific_force_g),
        ):
            history[key].append(value)  # type: ignore[union-attr]
        thrust_on_time_s += thrust_active_duration
        if altitude <= 0:
            break

    history["maximum_tilt_deg"] = maximum_tilt_deg
    history["landed"] = altitude <= 0
    history["thrust_on_time_s"] = thrust_on_time_s
    history["maximum_dynamic_pressure_pa"] = maximum_dynamic_pressure_pa
    history["maximum_specific_force_g"] = maximum_specific_force_g
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


class SatelliteFetchThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        latitude_bounds: tuple[float, float],
        longitude_bounds: tuple[float, float],
    ):
        super().__init__()
        self.latitude_bounds = latitude_bounds
        self.longitude_bounds = longitude_bounds

    def run(self) -> None:
        try:
            self.succeeded.emit(
                fetch_satellite_mosaic(
                    self.latitude_bounds,
                    self.longitude_bounds,
                )
            )
        except (requests.RequestException, OSError, ValueError) as error:
            self.failed.emit(str(error))


class CalculationThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(
        self,
        revision: int,
        calculation: Callable[[], dict[str, object]],
    ) -> None:
        super().__init__()
        self.revision = revision
        self.calculation = calculation

    def run(self) -> None:
        try:
            self.succeeded.emit((self.revision, self.calculation()))
        except ValueError as error:
            self.failed.emit((self.revision, str(error)))


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
        self.setWindowTitle("CanSat — Flight and Landing Simulator")
        self.resize(1500, 920)
        self.weather_profiles: list[WeatherProfile] = []
        self.weather_worker: WeatherFetchThread | None = None
        self.weather_location: tuple[float, float] | None = None
        self._pending_weather_refresh = False
        self._updating = False
        self.input_revision = 0
        self.map_pick_mode: str | None = None
        self._preserve_reach_on_update = False
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(180)
        self.update_timer.timeout.connect(self.update_all)
        self.calculation_worker: CalculationThread | None = None
        self.calculation_kind: str | None = None
        self.trajectory_colorbar = None
        self.forecast_colorbar = None
        self.reach_envelope: dict[str, object] | None = None
        self.navigation_plan: dict[str, object] | None = None
        self.satellite_worker: SatelliteFetchThread | None = None
        self.satellite_image: np.ndarray | None = None
        self.satellite_extent: tuple[float, float, float, float] | None = None
        self.satellite_status_text = ""
        self.satellite_request_key: tuple[float, ...] | None = None
        self.satellite_image_key: tuple[float, ...] | None = None
        self._satellite_refresh_pending = False
        self._pending_satellite_bounds: tuple[
            tuple[float, float], tuple[float, float]
        ] | None = None
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
        widget.valueChanged.connect(self._schedule_update)
        return widget

    def _schedule_update(
        self, *_args: object, preserve_reach: bool = False
    ) -> None:
        self.input_revision += 1
        self._preserve_reach_on_update = (
            self._preserve_reach_on_update or preserve_reach
        )
        if not preserve_reach:
            self.reach_envelope = None
        self.navigation_plan = None
        show_reach = getattr(self, "show_reach", None)
        if (
            not preserve_reach
            and show_reach is not None
            and show_reach.isChecked()
        ):
            show_reach.blockSignals(True)
            show_reach.setChecked(False)
            show_reach.blockSignals(False)
        self.update_timer.start()

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

        flight_box = QGroupBox("CanSat and Release Parameters")
        flight_form = QFormLayout(flight_box)
        self.mass = self._add_spin(flight_form, "Total mass", 0.05, 5, 0.35, " kg")
        self.release_height = self._add_spin(
            flight_form, "Release altitude", 10, 2500, 2300, " m", 0
        )
        self.canopy_area = self._add_spin(
            flight_form,
            "Canopy area (not volume)",
            50,
            50000,
            1100,
            " cm²",
            0,
            50,
        )
        self.shape = QComboBox()
        self.shape.addItems(SHAPES)
        self.shape.setCurrentText("Hemispherical")
        self.shape.currentTextChanged.connect(self._shape_changed)
        flight_form.addRow("Parachute type", self.shape)
        self.drag_coefficient = self._add_spin(
            flight_form, "Drag coefficient Cd", 0.2, 2.5, 1.3, "", 2, 0.05
        )
        self.canopy_inflation_time = self._add_spin(
            flight_form, "Canopy inflation time", 0, 20, 1.5, " s", 1, 0.1
        )
        self.thrust = self._add_spin(
            flight_form, "Total propulsion thrust (from motor data)", 0, 25, 0, " N", 2, 0.1
        )
        self.thrust.setReadOnly(False)
        self.thrust_duration = self._add_spin(
            flight_form, "Motor run time", 0, 300, 5, " s", 1, 1
        )
        self.heading = self._add_spin(
            flight_form, "Thrust azimuth (0° = north)", 0, 359, 90, "°", 0, 1
        )
        self.continuous_thrust = QCheckBox("Maintain thrust until landing")
        self.continuous_thrust.toggled.connect(self._thrust_mode_changed)
        flight_form.addRow(self.continuous_thrust)
        self.canopy_tilt_effect = QCheckBox(
            "Full drag vector includes the horizontal wind component"
        )
        self.canopy_tilt_effect.setChecked(True)
        self.canopy_tilt_effect.toggled.connect(self._schedule_update)
        flight_form.addRow(self.canopy_tilt_effect)
        self.integration_step = _spin(0.05, 0.5, 0.1, 2, 0.05)
        self.integration_step.setSuffix(" s")
        self.integration_step.valueChanged.connect(self._schedule_update)
        flight_form.addRow("Integration time step", self.integration_step)
        controls_layout.addWidget(flight_box)

        self.motor_data_box = QGroupBox("Motor, propeller, and energy data (optional)")
        motor_form = QFormLayout(self.motor_data_box)
        self.motor_count = QSpinBox()
        self.motor_count.setRange(1, 8)
        self.motor_count.setValue(2)
        motor_form.addRow("Number of motors", self.motor_count)
        self.motor_thrust = _spin(0, 25, 0, 2, 0.1)
        self.motor_thrust.setSuffix(" N per motor")
        motor_form.addRow("Rated motor thrust", self.motor_thrust)
        self.motor_voltage = _spin(1, 60, 7.4, 1, 0.1)
        self.motor_voltage.setSuffix(" V")
        motor_form.addRow("Voltage under load", self.motor_voltage)
        self.motor_current = _spin(0, 100, 3, 2, 0.1)
        self.motor_current.setSuffix(" A per motor")
        motor_form.addRow("Current while running", self.motor_current)
        self.propeller_diameter = _spin(1, 100, 5, 1, 0.5)
        self.propeller_diameter.setSuffix(" cm")
        motor_form.addRow("Propeller diameter", self.propeller_diameter)
        self.propeller_pitch = _spin(1, 50, 3, 1, 0.5)
        self.propeller_pitch.setSuffix(" cm")
        motor_form.addRow("Propeller pitch", self.propeller_pitch)
        self.motor_rpm = _spin(0, 100000, 0, 0, 100)
        self.motor_rpm.setSuffix(" rpm")
        motor_form.addRow("RPM under load (optional)", self.motor_rpm)
        self.battery_capacity = _spin(100, 20000, 2200, 0, 100)
        self.battery_capacity.setSuffix(" mAh")
        motor_form.addRow("Battery capacity", self.battery_capacity)
        self.motor_results = QLabel(
            "Enter values from the datasheet; energy is estimated as V × I × time."
        )
        self.motor_results.setWordWrap(True)
        motor_form.addRow(self.motor_results)
        for widget in (
            self.motor_count,
            self.motor_thrust,
            self.motor_voltage,
            self.motor_current,
            self.propeller_diameter,
            self.propeller_pitch,
            self.motor_rpm,
            self.battery_capacity,
        ):
            widget.valueChanged.connect(self._motor_configuration_changed)
        controls_layout.addWidget(self.motor_data_box)

        location_box = QGroupBox("Location and Operator Target")
        location_form = QFormLayout(location_box)
        self.latitude = _spin(-90, 90, 50.34, 5, 0.001)
        self.latitude.setSuffix("°")
        self.latitude.valueChanged.connect(self._schedule_update)
        self.pick_start_button = QPushButton("Pick on map")
        self.pick_start_button.setToolTip(
            "Click this, then click the launch location on the 2D terrain map."
        )
        self.pick_start_button.clicked.connect(
            lambda: self._begin_map_pick("start")
        )
        start_row = QWidget()
        start_layout = QHBoxLayout(start_row)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.addWidget(self.latitude, 1)
        start_layout.addWidget(self.pick_start_button)
        location_form.addRow("Launch latitude", start_row)
        self.longitude = self._add_spin(
            location_form, "Launch longitude", -180, 180, 19.51, "°", 5, 0.001
        )
        self.latitude.editingFinished.connect(self._location_changed)
        self.longitude.editingFinished.connect(self._location_changed)
        self.target_latitude = _spin(-90, 90, 50.34, 5, 0.001)
        self.target_latitude.setSuffix("°")
        self.target_latitude.valueChanged.connect(
            lambda: self._schedule_update(preserve_reach=True)
        )
        self.pick_target_button = QPushButton("Pick on map")
        self.pick_target_button.setToolTip(
            "Click this, then click the landing location on the 2D terrain map."
        )
        self.pick_target_button.clicked.connect(
            lambda: self._begin_map_pick("target")
        )
        target_row = QWidget()
        target_layout = QHBoxLayout(target_row)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.target_latitude, 1)
        target_layout.addWidget(self.pick_target_button)
        location_form.addRow("Target latitude", target_row)
        self.target_longitude = _spin(-180, 180, 19.51, 5, 0.001)
        self.target_longitude.setSuffix("°")
        self.target_longitude.valueChanged.connect(
            lambda: self._schedule_update(preserve_reach=True)
        )
        location_form.addRow("Target longitude", self.target_longitude)
        self.map_pick_status = QLabel(
            "To select a point, click “Pick on map” next to the launch or target coordinates."
        )
        self.map_pick_status.setWordWrap(True)
        location_form.addRow("Map selection", self.map_pick_status)
        controls_layout.addWidget(location_box)

        weather_box = QGroupBox("Weather — Open-Meteo by default")
        weather_form = QFormLayout(weather_box)
        self.weather_mode = QComboBox()
        self.weather_mode.addItems(("Open-Meteo forecast", "Manual weather"))
        self.weather_mode.currentTextChanged.connect(self._weather_mode_changed)
        weather_form.addRow("Data source", self.weather_mode)
        self.weather_model = QComboBox()
        self.weather_model.addItem("Automatic", "")
        self.weather_model.addItem("GFS Seamless", "gfs_seamless")
        weather_form.addRow("Forecast model", self.weather_model)
        self.weather_model.currentIndexChanged.connect(self._weather_model_changed)
        self.forecast_time = QComboBox()
        self.forecast_time.currentIndexChanged.connect(self._schedule_update)
        weather_form.addRow("Forecast time", self.forecast_time)
        self.fetch_button = QPushButton("Fetch / refresh forecast")
        self.fetch_button.clicked.connect(self.fetch_weather)
        weather_form.addRow(self.fetch_button)
        self.manual_wind = self._add_spin(
            weather_form, "Wind (constant)", 0, 60, 5, " m/s", 1, 0.5
        )
        self.manual_direction = self._add_spin(
            weather_form, "Wind direction (from)", 0, 359, 270, "°", 0, 1
        )
        self.manual_temperature = self._add_spin(
            weather_form, "Ground-level temperature", -60, 60, 15, " °C", 1, 0.5
        )
        self.manual_pressure = self._add_spin(
            weather_form, "Ground-level pressure", 300, 1100, 1000, " hPa", 1, 1
        )
        self.weather_status = QLabel("Waiting for Open-Meteo…")
        self.weather_status.setWordWrap(True)
        weather_form.addRow("Status", self.weather_status)
        controls_layout.addWidget(weather_box)

        measurements_box = QGroupBox("CanSat measurements for comparison with the forecast")
        measurements_form = QFormLayout(measurements_box)
        self.measurement_height = self._add_spin(
            measurements_form, "Measurement altitude", 0, 3000, 10, " m", 0, 1
        )
        self.measured_temperature = self._add_spin(
            measurements_form, "Measured temperature", -80, 80, 15, " °C", 1, 0.5
        )
        self.measured_pressure = self._add_spin(
            measurements_form, "Measured pressure", 100, 1200, 1000, " hPa", 1, 1
        )
        self.measured_wind_speed = self._add_spin(
            measurements_form, "Measured wind speed", 0, 100, 5, " m/s", 1, 0.5
        )
        self.measured_wind_direction = self._add_spin(
            measurements_form, "Measured wind direction (from)", 0, 359, 270, "°", 0, 1
        )
        controls_layout.addWidget(measurements_box)
        self.compare_result = QLabel("The comparison will appear after weather data is loaded.")
        self.compare_result.setWordWrap(True)
        controls_layout.addWidget(self.compare_result)
        controls_layout.addStretch(1)
        layout.addWidget(self._build_tabs(), 1)
        self._weather_mode_changed(self.weather_mode.currentText())

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        self.main_tabs = tabs

        trajectory_tab = QWidget()
        self.trajectory_tab = trajectory_tab
        trajectory_layout = QVBoxLayout(trajectory_tab)
        self.summary = QLabel("Calculating trajectory…")
        self.summary.setWordWrap(True)
        trajectory_layout.addWidget(self.summary)
        self.export_button = QPushButton("Export current trajectory to CSV")
        self.export_button.clicked.connect(self.export_trajectory)
        trajectory_layout.addWidget(self.export_button)
        operations_box = QGroupBox("Reach and Navigation")
        operations_layout = QHBoxLayout(operations_box)
        self.max_reach_thrust = _spin(0, 25, 0, 2, 0.1)
        self.max_reach_thrust.setSuffix(" N total")
        self.max_reach_thrust.valueChanged.connect(self._schedule_update)
        self.max_reach_thrust.setToolTip(
            "Defaults to the sum of thrust values from the motor configuration; override for analysis if needed."
        )
        operations_layout.addWidget(QLabel("Maximum thrust for reach and guidance:"))
        operations_layout.addWidget(self.max_reach_thrust)
        self.calculate_reach_button = QPushButton("Calculate reach")
        self.calculate_reach_button.clicked.connect(self.calculate_reach)
        operations_layout.addWidget(self.calculate_reach_button)
        self.show_reach = QCheckBox("Show reach envelope on plot")
        self.show_reach.toggled.connect(self._toggle_reach)
        operations_layout.addWidget(self.show_reach)
        self.navigate_button = QPushButton("Plan flight to target (±20 m)")
        self.navigate_button.clicked.connect(self.plan_navigation)
        operations_layout.addWidget(self.navigate_button)
        self.satellite_enabled = QCheckBox("Esri satellite map (online)")
        self.satellite_enabled.toggled.connect(self._toggle_satellite)
        operations_layout.addWidget(self.satellite_enabled)
        trajectory_layout.addWidget(operations_box)
        self.operation_status = QLabel(
            "Reach shows an asymmetric outline of landing locations under wind and thrust. "
            "Navigation minimizes landing error by changing heading and allowing "
            "the motors to be switched on and off multiple times."
        )
        self.operation_status.setWordWrap(True)
        trajectory_layout.addWidget(self.operation_status)
        self.satellite_status = QLabel(
            "The satellite image is displayed on a separate 2D map, without altitude distorting the map axes."
        )
        self.satellite_status.setWordWrap(True)
        trajectory_layout.addWidget(self.satellite_status)
        self.flight_figure = plt.figure(figsize=(10, 8))
        self.flight_figure.subplots_adjust(
            left=0.02, right=0.87, bottom=0.06, top=0.93
        )
        self.ax_3d = self.flight_figure.add_subplot(111, projection="3d")
        self.ax_3d.set_proj_type("ortho")
        self.flight_canvas = FigureCanvas(self.flight_figure)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(self.flight_canvas, "3D Trajectory")
        self.diagnostics_figure = plt.figure(figsize=(11, 8))
        self.diagnostics_figure.subplots_adjust(
            left=0.08, right=0.92, bottom=0.08, top=0.94, hspace=0.35, wspace=0.28
        )
        diagnostics_grid = self.diagnostics_figure.add_gridspec(2, 2)
        self.ax_altitude = self.diagnostics_figure.add_subplot(diagnostics_grid[0, 0])
        self.ax_speeds = self.diagnostics_figure.add_subplot(diagnostics_grid[0, 1])
        self.ax_tilt = self.diagnostics_figure.add_subplot(diagnostics_grid[1, 0])
        self.ax_navigation_control = self.diagnostics_figure.add_subplot(
            diagnostics_grid[1, 1]
        )
        self.ax_nav_heading = self.ax_navigation_control.twinx()
        self.diagnostics_canvas = FigureCanvas(self.diagnostics_figure)
        self.plot_tabs.addTab(self.diagnostics_canvas, "Time-series plots")
        self.map_figure = plt.figure(figsize=(9, 7))
        self.ax_ground_map = self.map_figure.add_subplot(111)
        self.map_canvas = FigureCanvas(self.map_figure)
        self.map_canvas.mpl_connect("button_press_event", self._map_clicked)
        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        map_layout.setContentsMargins(0, 0, 0, 0)
        self.map_toolbar = NavigationToolbar(self.map_canvas, map_tab)
        map_layout.addWidget(self.map_toolbar)
        map_layout.addWidget(self.map_canvas, 1)
        self.plot_tabs.addTab(map_tab, "2D terrain map")
        self.plot_tabs.currentChanged.connect(self._plot_tab_changed)
        trajectory_layout.addWidget(self.plot_tabs, 1)
        self.calculation_notes = QLabel(
            "Results use the parameters on the left, the selected weather profile, "
            "and the drag model. The forecast is not replaced by manually entered measurements."
        )
        self.calculation_notes.setWordWrap(True)
        trajectory_layout.addWidget(self.calculation_notes)
        tabs.addTab(trajectory_tab, "3D Flight")

        weather_tab = QWidget()
        weather_layout = QVBoxLayout(weather_tab)
        self.weather_tabs = QTabWidget()

        profile_tab = QWidget()
        profile_layout = QVBoxLayout(profile_tab)
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
        profile_layout.addWidget(self.weather_canvas, 1)
        self.profile_caption = QLabel(
            "Vertical profiles of wind, temperature, and pressure for the selected hour."
        )
        self.profile_caption.setWordWrap(True)
        profile_layout.addWidget(self.profile_caption)
        self.wind_direction_detail = QLabel(
            "Click a point on the wind direction plot to read the azimuth the wind is FROM."
        )
        self.wind_direction_detail.setWordWrap(True)
        profile_layout.addWidget(self.wind_direction_detail)
        self.weather_canvas.mpl_connect(
            "button_press_event", self._weather_plot_clicked
        )
        self.weather_tabs.addTab(profile_tab, "Vertical profile")

        forecast_tab = QWidget()
        forecast_layout = QVBoxLayout(forecast_tab)
        self.forecast_figure = plt.figure(figsize=(12, 7))
        forecast_grid = self.forecast_figure.add_gridspec(1, 3)
        self.ax_forecast_temperature = self.forecast_figure.add_subplot(
            forecast_grid[0, 0]
        )
        self.ax_forecast_wind = self.forecast_figure.add_subplot(forecast_grid[0, 1])
        self.ax_forecast_pressure = self.forecast_figure.add_subplot(
            forecast_grid[0, 2]
        )
        self.forecast_canvas = FigureCanvas(self.forecast_figure)
        forecast_layout.addWidget(self.forecast_canvas, 1)
        self.forecast_detail = QLabel(
            "Click the plot at a selected time and altitude to read "
            "wind (direction FROM), temperature, and pressure."
        )
        self.forecast_detail.setWordWrap(True)
        forecast_layout.addWidget(self.forecast_detail)
        self.forecast_canvas.mpl_connect(
            "button_press_event", self._forecast_plot_clicked
        )
        self.weather_tabs.addTab(forecast_tab, "Forecast: time and altitude")
        weather_layout.addWidget(self.weather_tabs, 1)
        tabs.addTab(weather_tab, "Weather")

        design_tab = QWidget()
        design_layout = QVBoxLayout(design_tab)
        design_box = QGroupBox("Parachute sizing")
        design_form = QFormLayout(design_box)
        self.parachute_mode = QComboBox()
        self.parachute_mode.addItems(
            ("Area for a specified speed", "Speed for a specified area")
        )
        self.parachute_mode.currentIndexChanged.connect(self._schedule_update)
        design_form.addRow("What should be calculated?", self.parachute_mode)
        self.target_descent_speed = _spin(0.5, 30, 5, 2, 0.1)
        self.target_descent_speed.setSuffix(" m/s")
        self.target_descent_speed.valueChanged.connect(self._schedule_update)
        design_form.addRow("Target descent speed", self.target_descent_speed)
        self.max_tilt = _spin(1, 75, 30, 1, 1)
        self.max_tilt.setSuffix("°")
        self.max_tilt.valueChanged.connect(self._schedule_update)
        design_form.addRow("Maximum acceptable tilt", self.max_tilt)
        self.design_result = QLabel()
        self.design_result.setWordWrap(True)
        design_form.addRow("Sizing result", self.design_result)
        self.apply_area_button = QPushButton("Apply calculated area to the simulation")
        self.apply_area_button.clicked.connect(self._apply_recommended_area)
        design_form.addRow(self.apply_area_button)
        design_layout.addWidget(design_box)

        self.shape_table = QTableWidget(4, 3)
        self.shape_table.setHorizontalHeaderLabels(
            ("Canopy type", "Area for target speed", "Equivalent diameter")
        )
        self.shape_table.horizontalHeader().setStretchLastSection(True)
        design_layout.addWidget(QLabel("Shape comparison at the specified speed"))
        design_layout.addWidget(self.shape_table)

        engine_box = QGroupBox("Motors and landing-site correction")
        engine_layout = QVBoxLayout(engine_box)
        self.engine_result = QLabel(
            "The tilt model is an approximation and does not establish a safe structural limit."
        )
        self.engine_result.setWordWrap(True)
        engine_layout.addWidget(self.engine_result)
        buttons = QHBoxLayout()
        self.thrust_limit_button = QPushButton("Calculate thrust limit")
        self.thrust_limit_button.clicked.connect(self.calculate_thrust_limit)
        buttons.addWidget(self.thrust_limit_button)
        self.correction_button = QPushButton("Find a correction toward the target")
        self.correction_button.clicked.connect(self.calculate_correction)
        buttons.addWidget(self.correction_button)
        engine_layout.addLayout(buttons)
        design_layout.addWidget(engine_box)
        design_layout.addStretch(1)
        tabs.addTab(design_tab, "Parachute and motor design")

        scenarios_tab = QWidget()
        scenarios_layout = QVBoxLayout(scenarios_tab)
        self.scenario_button = QPushButton("Run scenarios and uncertainty analysis")
        self.scenario_button.clicked.connect(self.run_scenarios)
        scenarios_layout.addWidget(self.scenario_button)
        self.ensemble_count = QSpinBox()
        self.ensemble_count.setRange(10, 100)
        self.ensemble_count.setValue(30)
        self.wind_uncertainty = _spin(0, 15, 1.5, 1, 0.5)
        self.wind_uncertainty.setSuffix(" m/s (1σ)")
        scenario_controls = QHBoxLayout()
        scenario_controls.addWidget(QLabel("Number of Monte Carlo trials:"))
        scenario_controls.addWidget(self.ensemble_count)
        scenario_controls.addWidget(QLabel("Wind-component uncertainty:"))
        scenario_controls.addWidget(self.wind_uncertainty)
        scenarios_layout.addLayout(scenario_controls)
        self.scenario_table = QTableWidget(0, 5)
        self.scenario_table.setHorizontalHeaderLabels(
            (
                "Scenario",
                "Time",
                "Distance to target",
                "Vertical speed",
                "Tilt",
            )
        )
        self.scenario_table.horizontalHeader().setStretchLastSection(True)
        scenarios_layout.addWidget(self.scenario_table)
        self.scenario_details = QLabel(
            "Running this will show exactly which parameters differ between "
            "variants and which input data were used."
        )
        self.scenario_details.setWordWrap(True)
        scenarios_layout.addWidget(self.scenario_details)
        self.scenario_figure = plt.figure(figsize=(8, 5))
        self.ax_scenarios = self.scenario_figure.add_subplot(111)
        self.scenario_canvas = FigureCanvas(self.scenario_figure)
        scenarios_layout.addWidget(self.scenario_canvas, 1)
        self.scenario_summary = QLabel(
            "Random trials are a sensitivity analysis, not a guarantee of in-flight dispersion."
        )
        self.scenario_summary.setWordWrap(True)
        scenarios_layout.addWidget(self.scenario_summary)
        tabs.addTab(scenarios_tab, "Scenarios")

        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        help_text = QLabel(
            "<h2>How to read the simulation</h2>"
            "<p>Set the release altitude from the rocket, launch coordinates and the "
            "operator's target point, CanSat mass, parachute parameters, and weather. "
            "The program recalculates the flight immediately; on first launch it fetches "
            "an Open-Meteo profile for the selected location.</p>"
            "<p><b>Choosing a location:</b> click <i>Pick on map</i> next to the launch "
            "latitude or target, then click a point on the <i>2D terrain map</i> tab. "
            "The reach outline does not block point selection. Pan/zoom tools are disabled "
            "automatically when selection starts. Clicking outside the map axes does not "
            "change the coordinates. Changing the launch point refreshes the Open-Meteo forecast.</p>"
            "<p><b>Canopy units:</b> enter the surface area in cm² (e.g. 1100 cm² = 0.11 m²), "
            "not cm³. The circular diameter shown in the results is an equivalent diameter; "
            "for other shapes, the actual projected area and measured Cd coefficient matter.</p>"
            "<p><b>3D flight:</b> the horizontal axes are WGS84 longitude and latitude, "
            "and the vertical axis is altitude above sea level. The green point marks release "
            "at the specified altitude, the red point marks the predicted landing, and the "
            "blue star marks the operator's target. The trajectory is converted from local "
            "east/north offsets in metres to geographic degrees.</p>"
            "<p><b>Weather:</b> choose a forecast hour or fixed manual conditions. Manual wind "
            "direction is the direction <i>from which</i> the wind blows. Manual pressure and "
            "temperature also determine air density.</p>"
            "<p><b>Design:</b> canopy sizing is based on the balance between weight and drag "
            "at steady state. The motor limit is a user-selected tilt threshold.</p>"
            "<p><b>Limitations:</b> the point-mass model does not include turbulence, canopy "
            "swing and deployment dynamics, tangled lines, or motor dynamics. The forecast is "
            "interpolated from pressure levels; it is not a measurement. Results do not replace "
            "hardware testing or a campaign safety assessment.</p>"
            "<p><b>Drag and tilt:</b> when enabled, the model calculates the drag vector "
            "relative to the air using all three velocity components. The comparison model, "
            "when disabled, calculates vertical drag as if the horizontal wind component had "
            "no effect on it. The airflow angle is estimated from air-relative velocity, but "
            "the canopy, lines, and CanSat are not separate bodies with pendulum dynamics.</p>"
            "<p><b>Scenarios:</b> compare motors off, current settings, wind strengthened/weakened "
            "by 2 m/s in the thrust direction, and a model with tilt coupling disabled. Monte Carlo "
            "runs use a random east/north wind offset. Details of the data and differences between "
            "variants appear below the table. The CSV button saves points from the current trajectory "
            "with position, velocity, tilt, and wind.</p>"
            "<p><b>Reach:</b> samples landing locations for different azimuths and durations of a "
            "single continuous burn. The directional outline interpolates the outer samples relative "
            "to the unpowered landing point and accounts for wind; it is an approximation, not a "
            "guarantee that every point can be reached.</p>"
            "<p><b>Navigation:</b> changes heading during one continuous flight, limiting the model's "
            "turn rate to 30°/s. It tests many ON/OFF schedules, so the motors may restart; it evaluates "
            "the predicted landing in wind and chooses the smallest error, leaving at least 15% of the "
            "flight for unpowered drift. The report shows the number of pulses, the final cutoff, the "
            "predicted landing, and the error relative to the target. This is an offline simulation plan, "
            "not a flight controller.</p>"
            "<p><b>Motors and energy (optional):</b> enable the propulsion model in Settings. Enter the "
            "number of motors, the thrust of one motor, voltage and current under load, propeller diameter "
            "and pitch, and optionally the RPM. Total propulsion thrust is the number of motors multiplied "
            "by the rated thrust. Energy is estimated as voltage × total current × ON time; the estimated "
            "mAh consumed, percentage of battery capacity, and propeller disk loading are also shown. This "
            "is a simplified energy balance, not a model of motor, ESC, or propeller characteristics.</p>"
            "<p><b>Satellite map:</b> an optional Esri image is shown on a separate 2D map in metres, together "
            "with the trajectory, target, and reach outline. It is not overlaid on the altitude axis of the 3D plot.</p>"
        )
        help_text.setWordWrap(True)
        help_text.setTextFormat(Qt.RichText)
        help_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        help_layout.addWidget(help_text)
        help_layout.addStretch(1)
        tabs.addTab(help_tab, "Help and assumptions")

        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_description = QLabel(
            "Enable only the submodels and views you need. "
            "When the motor model is disabled, total thrust can be entered manually."
        )
        settings_description.setWordWrap(True)
        settings_layout.addWidget(settings_description)
        self.use_motor_model = QCheckBox(
            "Use motor/propeller data and estimate energy consumption"
        )
        self.use_motor_model.setChecked(False)
        self.use_motor_model.toggled.connect(self._toggle_motor_model)
        settings_layout.addWidget(self.use_motor_model)
        self.show_motor_inputs = QCheckBox("Show propulsion data panel")
        self.show_motor_inputs.setChecked(True)
        self.show_motor_inputs.toggled.connect(self.motor_data_box.setVisible)
        settings_layout.addWidget(self.show_motor_inputs)
        self.show_forecast_view = QCheckBox(
            "Show time–altitude weather forecast"
        )
        self.show_forecast_view.setChecked(True)
        self.show_forecast_view.toggled.connect(
            lambda visible: self.weather_tabs.setTabVisible(1, visible)
        )
        settings_layout.addWidget(self.show_forecast_view)
        self.show_scenarios_view = QCheckBox("Show scenarios tab")
        self.show_scenarios_view.setChecked(True)
        self.show_scenarios_view.toggled.connect(
            lambda visible: tabs.setTabVisible(tabs.indexOf(scenarios_tab), visible)
        )
        settings_layout.addWidget(self.show_scenarios_view)
        self.show_control_chart = QCheckBox(
            "Show propulsion control schedule"
        )
        self.show_control_chart.setChecked(True)
        self.show_control_chart.toggled.connect(
            self.ax_navigation_control.set_visible
        )
        self.show_control_chart.toggled.connect(self.diagnostics_canvas.draw_idle)
        settings_layout.addWidget(self.show_control_chart)
        settings_layout.addWidget(self.satellite_enabled)
        settings_layout.addStretch(1)
        tabs.addTab(settings_tab, "Settings")
        self._toggle_motor_model(False)
        return tabs

    def _shape_changed(self, name: str) -> None:
        self.drag_coefficient.setValue(SHAPES[name])
        self._schedule_update()

    def _motor_configuration_changed(self) -> None:
        if not self.use_motor_model.isChecked():
            return
        combined_thrust = self.motor_count.value() * self.motor_thrust.value()
        self.thrust.blockSignals(True)
        self.thrust.setValue(combined_thrust)
        self.thrust.blockSignals(False)
        reach_control = getattr(self, "max_reach_thrust", None)
        if reach_control is not None:
            reach_control.setValue(combined_thrust)
        self._schedule_update()

    def _toggle_motor_model(self, enabled: bool) -> None:
        for widget in (
            self.motor_count,
            self.motor_thrust,
            self.motor_voltage,
            self.motor_current,
            self.propeller_diameter,
            self.propeller_pitch,
            self.motor_rpm,
            self.battery_capacity,
        ):
            widget.setEnabled(enabled)
        self.motor_results.setEnabled(enabled)
        self.thrust.setReadOnly(enabled)
        if enabled:
            self._motor_configuration_changed()
        else:
            self.navigation_plan = None
            self.motor_results.setText(
                "Propulsion model disabled. Enter total thrust manually; "
                "motor energy is not estimated."
            )
            self._schedule_update()

    def _thrust_mode_changed(self, checked: bool) -> None:
        self.thrust_duration.setEnabled(not checked)
        self._schedule_update()

    def _toggle_reach(self, _checked: bool) -> None:
        self._redraw_current_flight()

    def _toggle_satellite(self, checked: bool) -> None:
        if not checked:
            self.satellite_image = None
            self.satellite_extent = None
            self.satellite_image_key = None
            self.satellite_status.setText(
                "Satellite layer disabled; map tiles will not be fetched."
            )
        else:
            self.plot_tabs.setCurrentIndex(1)
            self.satellite_status.setText(
                "Fetching Esri World Imagery tiles for the visible area…"
            )
        self._redraw_current_flight()

    def _redraw_current_flight(self) -> None:
        flight = getattr(self, "current_flight", None)
        profile = getattr(self, "current_profile", None)
        if flight is not None and profile is not None:
            self._update_flight_plot(flight, profile)

    def _plot_tab_changed(self, index: int) -> None:
        if index == 2:
            self._redraw_current_flight()
            return
        canvas = self.plot_tabs.widget(index)
        if isinstance(canvas, FigureCanvas):
            canvas.draw_idle()

    def _begin_map_pick(self, mode: str) -> None:
        self._deactivate_map_navigation()
        self.map_pick_mode = mode
        self.map_pick_status.setText(
            "Click inside the 2D map to set the "
            f"{'launch point' if mode == 'start' else 'landing target'}. "
            "The reach outline does not block selection; clicking outside the map axes "
            "does not change the coordinates."
        )
        self.main_tabs.setCurrentWidget(self.trajectory_tab)
        self.plot_tabs.setCurrentIndex(2)

    def _deactivate_map_navigation(self) -> None:
        mode = getattr(self.map_toolbar.mode, "name", "")
        if mode == "PAN":
            self.map_toolbar.pan()
        elif mode == "ZOOM":
            self.map_toolbar.zoom()

    def _map_clicked(self, event: MouseEvent) -> None:
        if (
            self.map_pick_mode is None
            or event.inaxes is not self.ax_ground_map
            or event.button != 1
            or event.xdata is None
            or event.ydata is None
        ):
            return
        self._deactivate_map_navigation()
        latitude, longitude = local_to_geographic(
            float(event.xdata),
            float(event.ydata),
            self.latitude.value(),
            self.longitude.value(),
        )
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            self.map_pick_status.setText(
                "The selected point has invalid geographic coordinates."
            )
            return
        mode = self.map_pick_mode
        self.map_pick_mode = None
        if mode == "start":
            self.latitude.setValue(latitude)
            self.longitude.setValue(longitude)
            self.map_pick_status.setText(
                f"Launch set to {latitude:.6f}° N, {longitude:.6f}° E."
            )
            self._location_changed()
        else:
            self.target_latitude.setValue(latitude)
            self.target_longitude.setValue(longitude)
            self.map_pick_status.setText(
                f"Target set to {latitude:.6f}° N, {longitude:.6f}° E."
            )

    def _update_ground_map(
        self,
        profile: WeatherProfile,
        longitudes: list[float],
        latitudes: list[float],
        reach_east: list[float],
        reach_north: list[float],
    ) -> None:
        axis = self.ax_ground_map
        axis.clear()
        origin_latitude = self.latitude.value()
        origin_longitude = self.longitude.value()
        longitude_scale = 111320 * max(
            math.cos(math.radians(origin_latitude)), 0.01
        )
        if self.satellite_enabled.isChecked() and self.satellite_image is not None:
            lon_min, lon_max, lat_min, lat_max = self.satellite_extent
            image_extent = (
                (lon_min - origin_longitude) * longitude_scale,
                (lon_max - origin_longitude) * longitude_scale,
                (lat_min - origin_latitude) * 110540,
                (lat_max - origin_latitude) * 110540,
            )
            axis.imshow(
                self.satellite_image,
                extent=image_extent,
                origin="upper",
                interpolation="bilinear",
                zorder=0,
            )
            axis.text(
                0.01,
                0.01,
                "Esri World Imagery · Esri, Maxar, Earthstar Geographics",
                transform=axis.transAxes,
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
            )
        east = [(value - origin_longitude) * longitude_scale for value in longitudes]
        north = [(value - origin_latitude) * 110540 for value in latitudes]
        map_indices = _plot_indices(len(east), 2000)
        axis.plot(
            np.asarray(east)[map_indices],
            np.asarray(north)[map_indices],
            color="#00d1c1",
            linewidth=2,
            label="Trajektoria",
        )
        axis.scatter([0], [0], marker="^", color="lime", s=80, label="Start")
        axis.scatter(
            [east[-1]], [north[-1]], marker="o", color="red", s=55, label="Landing"
        )
        target_east, target_north = geographic_to_local(
            self.target_latitude.value(),
            self.target_longitude.value(),
            origin_latitude,
            origin_longitude,
        )
        axis.scatter(
            [target_east], [target_north], marker="*", color="yellow", s=110,
            label="Cel",
        )
        if reach_east and reach_north and self.reach_envelope is not None:
            boundary_east = reach_east + [reach_east[0]]
            boundary_north = reach_north + [reach_north[0]]
            axis.plot(
                boundary_east,
                boundary_north,
                color="#ff8c00",
                linewidth=2,
                marker="o" if len(boundary_east) <= 2 else None,
                label=(
                    "Unpowered landing"
                    if len(boundary_east) <= 2
                    else "Reach-sample outline (wind + propulsion)"
                ),
            )
        if self.navigation_plan is not None:
            nav_flight = self.navigation_plan["flight"]
            nav_east = self._value(nav_flight, "east")
            nav_north = self._value(nav_flight, "north")
            nav_thrust = self._value(nav_flight, "thrust_n")
            for threshold, color, label in (
                (True, "#ff4d00", "Propulsion ON"),
                (False, "#c084fc", "Propulsion OFF"),
            ):
                segments = [
                    [
                        (nav_east[index - 1], nav_north[index - 1]),
                        (nav_east[index], nav_north[index]),
                    ]
                    for index in range(1, len(nav_east))
                    if (nav_thrust[index] > 1e-6) == threshold
                ]
                if segments:
                    axis.add_collection(
                        LineCollection(
                            segments,
                            colors=color,
                            linewidths=1.7,
                            label=f"Plan: {label}",
                        )
                    )
            axis.autoscale_view()
        axis.set_title(
            "Satellite map and landing (top-down view)"
            if self.satellite_image is not None and self.satellite_enabled.isChecked()
            else f"Reach and trajectory map · terrain at {profile.elevation_m:.0f} m above sea level"
        )
        axis.set_xlabel("East of launch point (m)")
        axis.set_ylabel("North of launch point (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
        axis.margins(0.12)
        self.map_figure.tight_layout()
        self.map_canvas.draw_idle()

    def _update_map_without_weather(self) -> None:
        axis = self.ax_ground_map
        axis.clear()
        target_east, target_north = geographic_to_local(
            self.target_latitude.value(),
            self.target_longitude.value(),
            self.latitude.value(),
            self.longitude.value(),
        )
        east_min, east_max = sorted((0.0, target_east))
        north_min, north_max = sorted((0.0, target_north))
        east_padding = max((east_max - east_min) * 0.2, 500.0)
        north_padding = max((north_max - north_min) * 0.2, 500.0)
        axis.set_xlim(east_min - east_padding, east_max + east_padding)
        axis.set_ylim(north_min - north_padding, north_max + north_padding)
        axis.scatter([0], [0], marker="^", color="limegreen", s=80, label="Start")
        axis.scatter(
            [target_east], [target_north], marker="*", color="orange", s=110,
            label="Cel",
        )
        axis.set_title("Location selection map · local frame relative to launch")
        axis.set_xlabel("East of launch (m)")
        axis.set_ylabel("North of launch (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    def _request_satellite_tiles(
        self,
        latitude_bounds: tuple[float, float],
        longitude_bounds: tuple[float, float],
    ) -> None:
        if not self.satellite_enabled.isChecked():
            return
        request_key = tuple(
            round(value, 4)
            for value in (*latitude_bounds, *longitude_bounds)
        )
        if request_key == self.satellite_image_key:
            return
        if self.satellite_worker is not None and self.satellite_worker.isRunning():
            self._satellite_refresh_pending = True
            self._pending_satellite_bounds = (latitude_bounds, longitude_bounds)
            return
        self._satellite_refresh_pending = False
        self._pending_satellite_bounds = None
        worker = SatelliteFetchThread(latitude_bounds, longitude_bounds)
        worker.request_key = request_key
        worker.succeeded.connect(self._satellite_loaded)
        worker.failed.connect(self._satellite_failed)
        worker.finished.connect(self._satellite_finished)
        self.satellite_worker = worker
        self.satellite_request_key = request_key
        self.satellite_status.setText(
            "Downloading Esri World Imagery satellite tiles…"
        )
        worker.start()

    def _satellite_loaded(self, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 3:
            self._satellite_failed("The map server returned an invalid image.")
            return
        image, extent, zoom = result
        self.satellite_image = image
        self.satellite_extent = extent
        self.satellite_image_key = self.satellite_request_key
        self.satellite_status.setText(
            f"2D map: Esri World Imagery, zoom {zoom}, image up to 768×768 px. "
            "Source: Esri, Maxar, Earthstar Geographics. The 3D view is no longer "
            "covered with a stretched image."
        )
        self._redraw_current_flight()

    def _satellite_failed(self, message: str) -> None:
        self.satellite_status.setText(
            f"Could not download the satellite basemap: {message}"
        )

    def _satellite_finished(self) -> None:
        if self._satellite_refresh_pending and self._pending_satellite_bounds:
            bounds = self._pending_satellite_bounds
            self._satellite_refresh_pending = False
            self._pending_satellite_bounds = None
            self._request_satellite_tiles(*bounds)

    def _weather_model_changed(self) -> None:
        if self.weather_mode.currentText() == "Open-Meteo forecast":
            self.fetch_weather()

    def _location_changed(self) -> None:
        if self.weather_mode.currentText() == "Open-Meteo forecast":
            self.fetch_weather()
        else:
            self._schedule_update()

    def _weather_mode_changed(self, mode: str) -> None:
        is_forecast = mode == "Open-Meteo forecast"
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
                "Manual weather is active: constant wind and manually set "
                "temperature and pressure."
            )
        self._schedule_update()
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
            self._schedule_update()
        if self.weather_worker is not None and self.weather_worker.isRunning():
            self._pending_weather_refresh = True
            self.weather_status.setText(
                "The location or model changed; I will refresh the forecast after "
                "the current download finishes."
            )
            return
        self._pending_weather_refresh = False
        self.fetch_button.setEnabled(False)
        self.weather_status.setText("Fetching Open-Meteo forecast…")
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
            f"Open-Meteo forecast: {len(profiles)} profili godzinowych, "
            f"elevation {profiles[0].elevation_m:.0f} m above sea level; "
            f"location {location[0]:.5f}° N, {location[1]:.5f}° E; "
            f"model {self.weather_model.currentText()}."
        )
        if self.weather_mode.currentText() == "Open-Meteo forecast":
            self._schedule_update()
        else:
            self._update_forecast_plot()

    def _weather_failed(self, message: str) -> None:
        if self.weather_mode.currentText() == "Open-Meteo forecast":
            self.weather_profiles = []
            self.weather_location = None
            self.weather_status.setText(
                "Failed to fetch the forecast; it will not be silently replaced with "
                f"manual data. Select Manual weather or try again. Details: {message}"
            )
            self._schedule_update()
        else:
            self.weather_status.setText(
                "Forecast download failed; the active manual weather "
                f"remains unchanged. Details: {message}"
            )

    def _weather_finished(self) -> None:
        if self._pending_weather_refresh:
            self._pending_weather_refresh = False
            self.fetch_weather()
            return
        self.fetch_button.setEnabled(
            self.weather_mode.currentText() == "Open-Meteo forecast"
        )

    def _profile(self) -> WeatherProfile | None:
        if self.weather_mode.currentText() == "Manual weather":
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
            canopy_inflation_time_s=self.canopy_inflation_time.value(),
        )

    @staticmethod
    def _value(flight: dict[str, list[float] | float | bool], name: str) -> list[float]:
        return flight[name]  # type: ignore[return-value]

    def update_all(self, *_args: object) -> None:
        if self._updating:
            return
        self.update_timer.stop()
        preserve_reach = self._preserve_reach_on_update
        self._preserve_reach_on_update = False
        if not preserve_reach:
            self.reach_envelope = None
        self.navigation_plan = None
        if not preserve_reach and self.show_reach.isChecked():
            self.show_reach.blockSignals(True)
            self.show_reach.setChecked(False)
            self.show_reach.blockSignals(False)
        self._updating = True
        try:
            profile = self._profile()
            if profile is None:
                self.current_flight = None
                self.current_profile = None
                self.summary.setText(
                    "No weather profile. Wait for Open-Meteo or switch "
                    "to Manual weather."
                )
                self.compare_result.setText("No forecast is available for comparison with the measurement.")
                self.ax_3d.clear()
                self.ax_altitude.clear()
                self.ax_speeds.clear()
                self.ax_tilt.clear()
                self.ax_navigation_control.clear()
                self.ax_nav_heading.clear()
                self.ax_ground_map.clear()
                self.ax_wind.clear()
                self.ax_wind_direction.clear()
                self.ax_temperature.clear()
                self.ax_pressure.clear()
                self.ax_forecast_temperature.clear()
                self.ax_forecast_wind.clear()
                self.ax_forecast_pressure.clear()
                self.ax_ground_map.text(
                    0.5, 0.98, "Location selection map — no trajectory",
                    transform=self.ax_ground_map.transAxes, ha="center", va="top",
                )
                self._update_map_without_weather()
                self.ax_compass.clear()
                self.ax_compass.set_axis_off()
                self.ax_3d.text2D(
                    0.5,
                    0.5,
                    "Fetching Open-Meteo profile…",
                    transform=self.ax_3d.transAxes,
                    ha="center",
                )
                self.ax_altitude.text(
                    0.5, 0.5, "Waiting for data", transform=self.ax_altitude.transAxes,
                    ha="center", va="center"
                )
                self.ax_wind.text(
                    0.5, 0.5, "Waiting for data", transform=self.ax_wind.transAxes,
                    ha="center", va="center"
                )
                self.ax_wind_direction.text(
                    0.5,
                    0.5,
                    "Waiting for data",
                    transform=self.ax_wind_direction.transAxes,
                    ha="center",
                    va="center",
                )
                self.ax_forecast_temperature.text(
                    0.5,
                    0.5,
                    "Waiting for Open-Meteo forecast",
                    transform=self.ax_forecast_temperature.transAxes,
                    ha="center",
                    va="center",
                )
                self.flight_canvas.draw_idle()
                self.diagnostics_canvas.draw_idle()
                self.weather_canvas.draw_idle()
                self.forecast_canvas.draw_idle()
                self.map_canvas.draw_idle()
                return
            flight = self._flight(profile)
            if not flight["landed"]:
                raise ValueError("The simulation did not reach the ground within the 600 s limit.")
            self.current_flight = flight
            self.current_profile = profile
            self._update_flight_plot(flight, profile)
            self._update_weather_plot(profile)
            self._update_measurement_comparison(profile)
            self._update_design(profile, flight)
        except ValueError as error:
            self.summary.setText(f"Cannot calculate trajectory: {error}")
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
        reach_result = (
            self.reach_envelope
            if self.show_reach.isChecked() and self.reach_envelope is not None
            else None
        )
        reach_polygon_east: list[float] = []
        reach_polygon_north: list[float] = []
        reach_polygon_longitudes: list[float] = []
        reach_polygon_latitudes: list[float] = []
        if reach_result is not None:
            reach_boundary = reach_result["boundary"]
            reach_polygon_east = [point[0] for point in reach_boundary]
            reach_polygon_north = [point[1] for point in reach_boundary]
            reach_polygon_longitudes = [
                longitude + value / longitude_scale
                for value in reach_polygon_east
            ]
            reach_polygon_latitudes = [
                latitude + value / 110540 for value in reach_polygon_north
            ]
        navigation_flight = (
            self.navigation_plan["flight"]
            if self.navigation_plan is not None
            else None
        )
        navigation_east = (
            self._value(navigation_flight, "east")
            if navigation_flight is not None
            else []
        )
        navigation_north = (
            self._value(navigation_flight, "north")
            if navigation_flight is not None
            else []
        )
        navigation_altitude = (
            self._value(navigation_flight, "altitude")
            if navigation_flight is not None
            else []
        )
        navigation_longitudes = [
            longitude + value / longitude_scale for value in navigation_east
        ]
        navigation_latitudes = [
            latitude + value / 110540 for value in navigation_north
        ]
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
            f"<b>Scenario:</b> release at {self.release_height.value():.0f} m "
            f"AGL above {latitude:.5f}° N, {longitude:.5f}° E; mass "
            f"{self.mass.value():.3f} kg; canopy {self.canopy_area.value():.0f} cm² "
            f"(Cd={self.drag_coefficient.value():.2f}); thrust {self.thrust.value():.2f} N."
            f"<br><b>Flight time:</b> {times[-1]:.1f} s &nbsp; "
            f"<b>Vertical speed at ground level:</b> "
            f"{self._value(flight, 'speed_vertical')[-1]:.2f} m/s &nbsp; "
            f"<b>Horizontal:</b> {self._value(flight, 'speed_horizontal')[-1]:.2f} m/s"
            f"<br><b>Landing:</b> {latitudes[-1]:.6f}, {longitudes[-1]:.6f} "
            f"(WGS84, local approximation) &nbsp; "
            f"<b>Distance to target:</b> {distance:.1f} m &nbsp; "
            f"<b>Maximum tilt:</b> {maximum_tilt:.1f}° "
            f"(minimum vertical component of drag force "
            f"{min(vertical_factor):.2f})"
            f"<br><b>Loads:</b> maximum dynamic pressure "
            f"{float(flight['maximum_dynamic_pressure_pa']):.0f} Pa; "
            f"maximum acceleration from non-gravitational forces "
            f"{float(flight['maximum_specific_force_g']):.2f} g; "
            f"canopy inflation time {self.canopy_inflation_time.value():.1f} s "
            "(model CdA=CdAmax·(1−exp(−3t/T)))."
        )
        thrust_on_time = float(flight["thrust_on_time_s"])
        energy_wh = (
            self.motor_voltage.value()
            * self.motor_current.value()
            * self.motor_count.value()
            * thrust_on_time
            / 3600
        )
        used_capacity_mah = (
            self.motor_current.value()
            * self.motor_count.value()
            * thrust_on_time
            / 3.6
        )
        battery_used_percent = (
            100 * used_capacity_mah / self.battery_capacity.value()
        )
        power_w = (
            self.motor_voltage.value()
            * self.motor_current.value()
            * self.motor_count.value()
        )
        prop_radius_m = self.propeller_diameter.value() / 200
        propeller_diameter_m = self.propeller_diameter.value() / 100
        total_propeller_area = (
            self.motor_count.value() * math.pi * prop_radius_m**2
        )
        disk_loading = (
            self.thrust.value() / total_propeller_area
            if total_propeller_area > 0
            else 0.0
        )
        tip_speed_ms = math.pi * propeller_diameter_m * self.motor_rpm.value() / 60
        motor_summary = (
            f"Propulsion: {self.motor_count.value()} × {self.motor_thrust.value():.2f} N = "
            f"{self.thrust.value():.2f} N. Propeller {self.propeller_diameter.value():.1f} × "
            f"{self.propeller_pitch.value():.1f} cm; disk loading "
            f"{disk_loading:.1f} N/m²"
            + (
                f", blade tip speed {tip_speed_ms:.1f} m/s"
                if self.motor_rpm.value() > 0
                else ""
            )
            + (
                f". Total motor ON time: {thrust_on_time:.2f} s; energy "
                f"electrical approx. {energy_wh:.4f} Wh ({energy_wh * 1000:.1f} mWh) "
                f"at a power of {power_w:.1f} W, "
                f"charge used {used_capacity_mah:.1f} mAh ({battery_used_percent:.1f}% "
                f"of the battery capacity {self.battery_capacity.value():.0f} mAh). "
                f"Total propeller disk area: {total_propeller_area * 1e4:.1f} cm². "
                "The V×I×time energy estimate assumes constant rated current while running; "
                "it does not model the ESC, voltage drop, or "
                "motor/propeller characteristics."
            )
        )
        if self.use_motor_model.isChecked():
            self.motor_results.setText(motor_summary)
        else:
            self.motor_results.setText(
                "Propulsion model disabled. Enter total thrust manually; "
                "motor energy is not estimated."
            )
        if self.use_motor_model.isChecked() and self.navigation_plan is not None:
            planned_flight = self.navigation_plan["flight"]
            planned_on_time = float(planned_flight["thrust_on_time_s"])
            planned_energy_wh = power_w * planned_on_time / 3600
            planned_capacity_mah = (
                self.motor_current.value()
                * self.motor_count.value()
                * planned_on_time
                / 3.6
            )
            self.motor_results.setText(
                self.motor_results.text()
                + f" Target plan: {planned_on_time:.2f} s ON, "
                f"{planned_energy_wh:.4f} Wh and {planned_capacity_mah:.1f} mAh."
            )
        self.calculation_notes.setText(
            f"<b>Data used:</b> {profile.timestamp}, source "
            f"{'Open-Meteo ' + self.weather_model.currentText() if self.weather_mode.currentText() == 'Open-Meteo forecast' else 'manual weather'}; "
            f"temperature, pressure, and east/north wind vector interpolated from the profile "
            f"every 50 m (pressure levels 975–500 hPa at 25 hPa intervals). Launch "
            f"{latitude:.5f}° N, {longitude:.5f}° E; altitude "
            f"{self.release_height.value():.0f} m; mass {self.mass.value():.3f} kg; "
            f"canopy {self.canopy_area.value():.0f} cm², Cd={self.drag_coefficient.value():.2f}; "
            f"thrust {self.thrust.value():.2f} N; motor ON time "
            f"{thrust_on_time:.2f} s; estimated energy "
            f"{energy_wh:.4f} Wh. <b>Calculation:</b> integration of drag and gravity forces "
            f"every {self.integration_step.value():.2f} s until reaching the "
            "ground. Measurements entered in the panel are for comparison only; they do not change "
            "the forecast or simulation."
        )
        self.ax_3d.clear()
        map_longitudes = (
            longitudes
            + reach_polygon_longitudes
            + navigation_longitudes
            + [self.target_longitude.value()]
        )
        map_latitudes = (
            latitudes
            + reach_polygon_latitudes
            + navigation_latitudes
            + [self.target_latitude.value()]
        )
        map_lon_min, map_lon_max = min(map_longitudes), max(map_longitudes)
        map_lat_min, map_lat_max = min(map_latitudes), max(map_latitudes)
        map_x_span = max((map_lon_max - map_lon_min) * longitude_scale, 20.0)
        map_y_span = max((map_lat_max - map_lat_min) * 110540, 20.0)
        map_lon_bounds = (
            map_lon_min - map_x_span * 0.15 / longitude_scale,
            map_lon_max + map_x_span * 0.15 / longitude_scale,
        )
        map_lat_bounds = (
            map_lat_min - map_y_span * 0.15 / 110540,
            map_lat_max + map_y_span * 0.15 / 110540,
        )
        focus_east = [0.0, *east, *reach_polygon_east, *navigation_east]
        focus_north = [0.0, *north, *reach_polygon_north, *navigation_north]
        focus_e_min, focus_e_max = min(focus_east), max(focus_east)
        focus_n_min, focus_n_max = min(focus_north), max(focus_north)
        x_span = max(focus_e_max - focus_e_min, 20.0)
        y_span = max(focus_n_max - focus_n_min, 20.0)
        x_padding, y_padding = max(x_span * 0.12, 10.0), max(y_span * 0.12, 10.0)
        x_min, x_max = focus_e_min - x_padding, focus_e_max + x_padding
        y_min, y_max = focus_n_min - y_padding, focus_n_max + y_padding
        x_span, y_span = x_max - x_min, y_max - y_min
        plot_indices = _plot_indices(len(times))
        points = self.ax_3d.scatter(
            np.asarray(longitudes)[plot_indices],
            np.asarray(latitudes)[plot_indices],
            np.asarray(altitude_asl)[plot_indices],
            c=np.asarray(altitude)[plot_indices],
            cmap="viridis",
            s=9,
        )
        self.ax_3d.plot(
            np.asarray(longitudes)[plot_indices],
            np.asarray(latitudes)[plot_indices],
            np.asarray(altitude_asl)[plot_indices],
            color="#087e8b",
            alpha=0.75,
            linewidth=1.8,
        )
        if reach_result is not None:
            reach_points = np.asarray(reach_result["points"], dtype=float)
            reach_indices = _plot_indices(len(reach_points), 300)
            if len(reach_polygon_east) > 2:
                hull_closed_lons = reach_polygon_longitudes + [reach_polygon_longitudes[0]]
                hull_closed_lats = reach_polygon_latitudes + [reach_polygon_latitudes[0]]
                self.ax_3d.plot(
                    hull_closed_lons,
                    hull_closed_lats,
                    [profile.elevation_m] * len(hull_closed_lons),
                    color="#d97706",
                    linestyle="-",
                    linewidth=2,
                    label="Directional outline of reach samples",
                )
                self.ax_3d.scatter(
                    longitude + reach_points[reach_indices, 0] / longitude_scale,
                    latitude + reach_points[reach_indices, 1] / 110540,
                    [profile.elevation_m] * len(reach_indices),
                    color="#f4a261",
                    s=5,
                    alpha=0.4,
                    label="Landing samples",
                )
            else:
                self.ax_3d.scatter(
                    [reach_polygon_longitudes[0]],
                    [reach_polygon_latitudes[0]],
                    [profile.elevation_m],
                    color="#d97706",
                    marker="D",
                    s=45,
                    label="Unpowered landing (no area)",
                )
        if navigation_flight is not None:
            nav_altitude_asl = [
                profile.elevation_m + value for value in navigation_altitude
            ]
            nav_thrust = self._value(navigation_flight, "thrust_n")
            powered_segments = [
                [
                    (navigation_longitudes[index - 1], navigation_latitudes[index - 1], nav_altitude_asl[index - 1]),
                    (navigation_longitudes[index], navigation_latitudes[index], nav_altitude_asl[index]),
                ]
                for index in range(1, len(nav_thrust))
                if nav_thrust[index] > 1e-6
            ]
            coast_segments = [
                [
                    (navigation_longitudes[index - 1], navigation_latitudes[index - 1], nav_altitude_asl[index - 1]),
                    (navigation_longitudes[index], navigation_latitudes[index], nav_altitude_asl[index]),
                ]
                for index in range(1, len(nav_thrust))
                if nav_thrust[index] <= 1e-6
            ]
            if powered_segments:
                self.ax_3d.add_collection3d(
                    Line3DCollection(
                        powered_segments,
                        colors="#f97316",
                        linewidths=2,
                        label="Guidance: motors ON",
                    )
                )
            if coast_segments:
                self.ax_3d.add_collection3d(
                    Line3DCollection(
                        coast_segments,
                        colors="#7c3aed",
                        linewidths=1.5,
                        linestyles="dashed",
                        label="Guidance: motors OFF",
                    )
                )
        if self.trajectory_colorbar is None:
            self.trajectory_colorbar = self.flight_figure.colorbar(
                points,
                ax=self.ax_3d,
                shrink=0.55,
                pad=0.12,
                label="Altitude AGL (m)",
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
            label="Release",
        )
        self.ax_3d.scatter(
            [longitudes[-1]], [latitudes[-1]], [profile.elevation_m],
            color="red", marker="o", label="Landing"
        )
        self.ax_3d.set_title("3D trajectory — geographic coordinates")
        self.ax_3d.set_xlabel("Longitude (°E)")
        self.ax_3d.set_ylabel("Latitude (°N)")
        self.ax_3d.set_zlabel("Altitude above sea level (m)")
        z_min = max(profile.elevation_m - 1, -1)
        z_max = profile.elevation_m + self.release_height.value()
        z_span = max(z_max - z_min, 20.0)
        self.ax_3d.set_xlim(
            longitude + x_min / longitude_scale,
            longitude + x_max / longitude_scale,
        )
        self.ax_3d.set_ylim(
            latitude + y_min / 110540,
            latitude + y_max / 110540,
        )
        self.ax_3d.set_zlim(z_min, z_max)
        self.ax_3d.set_box_aspect((x_span, y_span, z_span))
        self.ax_3d.view_init(elev=24, azim=-55)
        target_is_visible = (
            x_min <= target_east <= x_max and y_min <= target_north <= y_max
        )
        if target_is_visible:
            self.ax_3d.scatter(
                [self.target_longitude.value()],
                [self.target_latitude.value()],
                [profile.elevation_m],
                color="blue",
                marker="*",
                s=100,
                label="Operator target",
            )
        else:
            self.ax_3d.text2D(
                0.02,
                0.97,
                "Target outside 3D frame — full view on the 2D terrain map",
                transform=self.ax_3d.transAxes,
                va="top",
                fontsize=9,
                color="#1d4ed8",
            )
        self.ax_3d.legend(fontsize=8, loc="upper left")
        if self.satellite_enabled.isChecked():
            self._request_satellite_tiles(map_lat_bounds, map_lon_bounds)
        if self.plot_tabs.currentIndex() == 2:
            self._update_ground_map(
                profile,
                longitudes,
                latitudes,
                reach_polygon_east,
                reach_polygon_north,
            )

        self.ax_altitude.clear()
        self.ax_altitude.plot(
            np.asarray(times)[plot_indices],
            np.asarray(altitude)[plot_indices],
            color="#264653",
        )
        self.ax_altitude.set_title("Altitude above terrain")
        self.ax_altitude.set_xlabel("Time (s)")
        self.ax_altitude.set_ylabel("Altitude (m)")
        self.ax_altitude.grid(True, alpha=0.3)

        self.ax_speeds.clear()
        self.ax_speeds.plot(
            np.asarray(times)[plot_indices],
            np.asarray(self._value(flight, "speed_horizontal"))[plot_indices],
            color="#087e8b",
            label="Horizontal relative to ground",
        )
        self.ax_speeds.plot(
            np.asarray(times)[plot_indices],
            np.asarray(self._value(flight, "speed_vertical"))[plot_indices],
            color="#e76f51",
            label="Vertical",
        )
        self.ax_speeds.set_title("CanSat speed")
        self.ax_speeds.set_xlabel("Time (s)")
        self.ax_speeds.set_ylabel("m/s")
        self.ax_speeds.legend(fontsize=8)
        self.ax_speeds.grid(True, alpha=0.3)

        self.ax_tilt.clear()
        self.ax_tilt.plot(
            np.asarray(times)[plot_indices],
            np.asarray(tilt)[plot_indices],
            color="#e76f51",
        )
        self.ax_tilt.axhline(self.max_tilt.value(), color="black", linestyle="--")
        self.ax_tilt.set_title("Estimated canopy tilt")
        self.ax_tilt.set_xlabel("Time (s)")
        self.ax_tilt.set_ylabel("Angle (°)")
        self.ax_tilt.grid(True, alpha=0.3)
        self.ax_tilt.text(
            0.01,
            0.98,
            (
                "Tilt effect: cos(angle) × vertical drag"
                if self.canopy_tilt_effect.isChecked()
                else "Tilt effect on vertical drag disabled"
            ),
            transform=self.ax_tilt.transAxes,
            va="top",
            fontsize=8,
        )
        self.ax_navigation_control.clear()
        self.ax_nav_heading.clear()
        heading_display: np.ndarray | None = None
        self.ax_navigation_control.step(
            times,
            self._value(flight, "thrust_n"),
            where="post",
            color="#087e8b",
            label="Current simulation",
        )
        if self.navigation_plan is not None:
            planned_flight = self.navigation_plan["flight"]
            planned_times = self._value(planned_flight, "time")
            self.ax_navigation_control.step(
                planned_times,
                self._value(planned_flight, "thrust_n"),
                where="post",
                color="#f97316",
                label="Plan: ON/OFF",
            )
            planned_heading = self._value(
                planned_flight, "thrust_heading_deg"
            )
            on_indices = [
                index
                for index, value in enumerate(
                    self._value(planned_flight, "thrust_n")
                )
                if value > 1e-6
            ]
            heading_display = np.degrees(
                np.unwrap(
                    np.radians(
                        [planned_heading[index] for index in on_indices]
                    )
                )
            )
            self.ax_nav_heading.plot(
                [planned_times[index] for index in on_indices],
                heading_display,
                color="#7c3aed",
                alpha=0.75,
                linewidth=1,
                label="Thrust direction",
            )
        self.ax_navigation_control.set_title("Propulsion control")
        self.ax_navigation_control.set_xlabel("Time (s)")
        self.ax_navigation_control.set_ylabel("Thrust (N)")
        self.ax_navigation_control.set_ylim(bottom=0)
        self.ax_nav_heading.set_ylabel("Thrust azimuth (°; continuous)")
        if heading_display is not None and len(heading_display):
            heading_min, heading_max = float(min(heading_display)), float(max(heading_display))
            heading_padding = max((heading_max - heading_min) * 0.1, 5.0)
            self.ax_nav_heading.set_ylim(
                heading_min - heading_padding, heading_max + heading_padding
            )
        else:
            self.ax_nav_heading.set_ylim(0, 360)
        self.ax_navigation_control.grid(True, alpha=0.3)
        control_handles, control_labels = self.ax_navigation_control.get_legend_handles_labels()
        heading_handles, heading_labels = self.ax_nav_heading.get_legend_handles_labels()
        if control_handles or heading_handles:
            self.ax_navigation_control.legend(
                control_handles + heading_handles,
                control_labels + heading_labels,
                fontsize=7,
                loc="upper right",
            )
        if self.plot_tabs.currentIndex() == 0:
            self.flight_canvas.draw_idle()
        elif self.plot_tabs.currentIndex() == 1:
            self.diagnostics_canvas.draw_idle()

    def calculate_reach(self) -> None:
        profile = self._profile()
        if profile is None:
            self.operation_status.setText(
                "No weather profile. Fetch a forecast or select Manual weather."
            )
            return
        max_thrust = self.max_reach_thrust.value()
        parameters = (
            self.mass.value(),
            self.canopy_area.value() / 10_000,
            self.drag_coefficient.value(),
            self.release_height.value(),
            max_thrust,
            self.heading.value(),
            profile,
        )
        self.operation_status.setText(
            f"Calculating landing locations for sampled headings and thrust durations "
            f"do {max_thrust:.2f} N w tle…"
        )
        self._start_calculation(
            "reach",
            lambda: compute_reach_envelope(
                *parameters,
                time_step_s=1.5,
                heading_count=24,
                burn_samples=9,
                canopy_inflation_time_s=self.canopy_inflation_time.value(),
            ),
        )

    def plan_navigation(self) -> None:
        profile = self._profile()
        if profile is None:
            self.operation_status.setText(
                "No weather profile. Fetch a forecast or select Manual weather."
            )
            return
        max_thrust = self.max_reach_thrust.value()
        target_east, target_north = geographic_to_local(
            self.target_latitude.value(),
            self.target_longitude.value(),
            self.latitude.value(),
            self.longitude.value(),
        )
        parameters = (
            self.mass.value(),
            self.canopy_area.value() / 10_000,
            self.drag_coefficient.value(),
            self.release_height.value(),
            max_thrust,
            target_east,
            target_north,
            self.heading.value(),
            profile,
        )
        self.operation_status.setText(
            f"Searching for a motor schedule at a thrust of {max_thrust:.2f} N. "
            "Heading may change and the motor may switch on and off multiple times; "
            "calculations are running in the background…"
        )
        self._start_calculation(
            "navigation",
            lambda: plan_navigation_to_target(
                *parameters,
                time_step_s=0.5,
                tolerance_m=20.0,
                canopy_inflation_time_s=self.canopy_inflation_time.value(),
            ),
        )

    def _start_calculation(
        self,
        kind: str,
        calculation: Callable[[], dict[str, object]],
    ) -> None:
        if self.calculation_worker is not None:
            self.operation_status.setText(
                "Another calculation is already running. Wait for it to finish."
            )
            return
        self.update_timer.stop()
        self.update_all()
        self.calculation_kind = kind
        if kind == "reach":
            self.calculate_reach_button.setEnabled(False)
        else:
            self.navigate_button.setEnabled(False)
        worker = CalculationThread(self.input_revision, calculation)
        worker.succeeded.connect(self._calculation_succeeded)
        worker.failed.connect(self._calculation_failed)
        worker.finished.connect(self._calculation_finished)
        self.calculation_worker = worker
        worker.start()

    def _calculation_succeeded(self, payload: object) -> None:
        revision, result = payload
        if revision != self.input_revision:
            self.operation_status.setText(
                "Parameters changed during calculation; the stale result "
                "was discarded. Run the calculation again."
            )
            return
        if self.calculation_kind == "reach":
            self._apply_reach_result(result)
        else:
            self._apply_navigation_result(result)

    def _calculation_failed(self, payload: object) -> None:
        revision, message = payload
        if revision == self.input_revision:
            operation = (
                "reach" if self.calculation_kind == "reach" else "navigation"
            )
            self.operation_status.setText(
                f"Failed to calculate {operation}: {message}"
            )

    def _calculation_finished(self) -> None:
        if self.calculation_kind == "reach":
            self.calculate_reach_button.setEnabled(True)
        else:
            self.navigate_button.setEnabled(True)
        worker = self.calculation_worker
        self.calculation_worker = None
        self.calculation_kind = None
        if worker is not None:
            worker.deleteLater()

    def _apply_reach_result(self, result: dict[str, object]) -> None:
        self.reach_envelope = result
        self.show_reach.setChecked(True)
        self._redraw_current_flight()
        profile = self.current_profile
        self.operation_status.setText(
            f"Samples: {result['samples']} landing locations, thrust up to "
            f"{result['max_thrust_n']:.2f} N, weather {profile.timestamp}. "
            "The orange line is a directional interpolation of outer "
            "samples, not a guaranteed reachability boundary."
        )

    def _apply_navigation_result(self, result: dict[str, object]) -> None:
        self.navigation_plan = result
        self._redraw_current_flight()
        flight = result["flight"]
        landing_east = float(self._value(flight, "east")[-1])
        landing_north = float(self._value(flight, "north")[-1])
        landing_latitude = self.latitude.value() + landing_north / 110540
        landing_longitude = self.longitude.value() + landing_east / (
            111320 * max(math.cos(math.radians(self.latitude.value())), 0.01)
        )
        tolerance_status = (
            "TARGET REACHED WITHIN ±20 m"
            if result["within_tolerance"]
            else "±20 m tolerance NOT MET"
        )
        energy_wh = (
            self.motor_voltage.value()
            * self.motor_current.value()
            * self.motor_count.value()
            * float(flight["thrust_on_time_s"])
            / 3600
            if self.use_motor_model.isChecked()
            else None
        )
        cutoff_latitude = (
            self.latitude.value() + result["cutoff_north_m"] / 110540
        )
        cutoff_longitude = self.longitude.value() + result["cutoff_east_m"] / (
            111320 * max(math.cos(math.radians(self.latitude.value())), 0.01)
        )
        energy_text = (
            f" Energy: {energy_wh:.4f} Wh."
            if energy_wh is not None
            else " Energy model disabled."
        )
        self.operation_status.setText(
            f"{tolerance_status}: error {result['miss_distance_m']:.1f} m. "
            f"Propulsion: {result['engine_pulses']} pulses, initial azimuth "
            f"{result['heading_deg']:.1f}°, final cutoff after "
            f"{result['cutoff_time_s']:.1f} s at an altitude of "
            f"{result['cutoff_altitude_m']:.0f} m AGL "
            f"({cutoff_latitude:.6f}° N, {cutoff_longitude:.6f}° E); "
            f"total ON time {result['burn_duration_s']:.1f} s. Heading "
            f"{'changed during flight' if result['heading_changes_during_flight'] else 'remained constant'}. "
            "Landing "
            f"{landing_latitude:.6f}° N, {landing_longitude:.6f}° E. "
            f"Maximum thrust {result['max_thrust_n']:.2f} N, ON time "
            f"{result['thrust_on_time_s']:.1f} s; weather {self.current_profile.timestamp}."
            f"{energy_text} Orange: propulsion ON; purple: unpowered drift. "
            "This is a simplified simulation, not a guarantee of safety or target hit."
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if (
            self.calculation_worker is not None
            and self.calculation_worker.isRunning()
        ):
            self.calculation_worker.wait()
        super().closeEvent(event)

    def _update_weather_plot(self, profile: WeatherProfile) -> None:
        self.ax_wind.clear()
        self.ax_wind_direction.clear()
        self.ax_temperature.clear()
        self.ax_pressure.clear()
        wind_speed = np.hypot(profile.wind_east_ms, profile.wind_north_ms)
        self.ax_wind.plot(wind_speed, profile.heights_m, color="#087e8b")
        self.ax_wind.set_title("Wind")
        self.ax_wind.set_xlabel("m/s")
        self.ax_wind.set_ylabel("Height above ground (m)")
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
        self.ax_wind_direction.set_title("Wind direction (FROM)")
        self.ax_wind_direction.set_xlabel("Meteorological azimuth")
        self.ax_wind_direction.set_ylabel("Height above ground (m)")
        sample_height = min(self.measurement_height.value(), float(profile.heights_m[-1]))
        self.ax_wind_direction.axhline(sample_height, color="#e76f51", linestyle="--")
        self.ax_temperature.plot(profile.temperature_c, profile.heights_m, color="#e76f51")
        self.ax_temperature.set_title("Temperature")
        self.ax_temperature.set_xlabel("°C")
        self.ax_pressure.plot(profile.pressure_hpa, profile.heights_m, color="#264653")
        self.ax_pressure.set_title("Pressure")
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
            f"Wind at {sample_height:.0f} m\narrow points FROM",
            pad=18,
        )
        sample_east, sample_north, _, _, _ = profile.at(sample_height)
        compass_bearing = wind_direction_from_components(sample_east, sample_north)
        if compass_bearing is None:
            self.ax_compass.text(
                0.5, 0.5, "Calm\nwind", transform=self.ax_compass.transAxes,
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
                f"FROM {compass_bearing:.0f}° ({wind_speed[np.argmin(abs(profile.heights_m - sample_height))]:.1f} m/s)\n"
                f"TOWARD {(compass_bearing + 180) % 360:.0f}°",
                transform=self.ax_compass.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
        self.weather_figure.tight_layout()
        self.weather_canvas.draw_idle()
        self.profile_caption.setText(
            f"Profile: {profile.timestamp}. Direction indicates where the wind comes FROM "
            "(0°=N, 90°=E); the blue arrow points TOWARD the direction the wind blows. "
            "Open-Meteo provides temperature_2m, wind_speed_10m, wind_direction_10m, "
            "surface_pressure and, at pressure levels 975–500 hPa in 25 hPa increments, "
            "temperature, wind speed and direction, and geopotential height. Values are "
            "interpolated every 50 m. Manual weather assumes constant wind."
        )
        self.wind_direction_detail.setText(
            "Click a point on the wind direction plot to read the azimuth the wind is FROM."
        )
        self._update_forecast_plot()

    def _weather_plot_clicked(self, event: object) -> None:
        if (
            getattr(event, "inaxes", None) is not self.ax_wind_direction
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return
        profile = getattr(self, "current_profile", None)
        if profile is None:
            return
        directions = np.asarray(
            [
                wind_direction_from_components(east, north)
                if math.hypot(east, north) >= 1e-6
                else np.nan
                for east, north in zip(profile.wind_east_ms, profile.wind_north_ms)
            ],
            dtype=float,
        )
        valid_indices = np.flatnonzero(np.isfinite(directions))
        if not len(valid_indices):
            self.wind_direction_detail.setText(
                f"{profile.timestamp}: calm wind — direction undefined."
            )
            return
        heights = np.asarray(profile.heights_m, dtype=float)
        angular_delta = (
            (directions[valid_indices] - float(event.xdata) + 180) % 360 - 180
        ) / 360
        height_scale = max(float(np.ptp(heights)), 1.0)
        height_delta = (
            heights[valid_indices] - float(event.ydata)
        ) / height_scale
        index = int(valid_indices[np.argmin(angular_delta**2 + height_delta**2)])
        speed = math.hypot(
            float(profile.wind_east_ms[index]),
            float(profile.wind_north_ms[index]),
        )
        self.wind_direction_detail.setText(
            f"Selected point: {profile.timestamp}, altitude "
            f"{heights[index]:.0f} m above ground; wind comes FROM "
            f"{directions[index]:.1f}° (0°=N, 90°=E), "
            f"{speed:.1f} m/s."
        )

    def _update_forecast_plot(self) -> None:
        if not self.weather_profiles:
            if self.forecast_colorbar is not None:
                colorbar_axis = self.forecast_colorbar.ax
                if colorbar_axis in self.forecast_figure.axes:
                    self.forecast_figure.delaxes(colorbar_axis)
                self.forecast_colorbar = None
            self.ax_forecast_temperature.clear()
            self.ax_forecast_wind.clear()
            self.ax_forecast_pressure.clear()
            self.ax_forecast_temperature.text(
                0.5,
                0.5,
                "Select an Open-Meteo forecast",
                transform=self.ax_forecast_temperature.transAxes,
                ha="center",
                va="center",
            )
            self.forecast_canvas.draw_idle()
            return
        heights = self.weather_profiles[0].heights_m
        times = [
            datetime.fromisoformat(profile.timestamp)
            for profile in self.weather_profiles
        ]
        time_values = mdates.date2num(times)
        time_grid, height_grid = np.meshgrid(time_values, heights)
        temperatures = np.column_stack(
            [profile.temperature_c for profile in self.weather_profiles]
        )
        pressures = np.column_stack(
            [profile.pressure_hpa for profile in self.weather_profiles]
        )
        wind_east = np.column_stack(
            [profile.wind_east_ms for profile in self.weather_profiles]
        )
        wind_north = np.column_stack(
            [profile.wind_north_ms for profile in self.weather_profiles]
        )

        self.ax_forecast_temperature.clear()
        self.ax_forecast_wind.clear()
        self.ax_forecast_pressure.clear()
        temperature_plot = self.ax_forecast_temperature.contourf(
            time_grid,
            height_grid,
            temperatures,
            levels=20,
            cmap="RdYlBu_r",
        )
        if self.forecast_colorbar is None:
            self.forecast_colorbar = self.forecast_figure.colorbar(
                temperature_plot,
                ax=self.ax_forecast_temperature,
                label="Temperature (°C)",
            )
        else:
            self.forecast_colorbar.update_normal(temperature_plot)
        self.ax_forecast_temperature.set_title("Temperature: time × altitude")
        self.ax_forecast_temperature.set_ylabel("Height above ground (m)")
        self.ax_forecast_wind.quiver(
            time_grid[::2, ::2],
            height_grid[::2, ::2],
            wind_east[::2, ::2],
            wind_north[::2, ::2],
            np.hypot(wind_east[::2, ::2], wind_north[::2, ::2]),
            cmap="viridis",
            scale=100,
        )
        self.ax_forecast_wind.set_title("Wind (arrows: E/N vector)")
        self.ax_forecast_wind.set_yticklabels([])
        self.ax_forecast_pressure.set_title("Pressure (hPa)")
        self.ax_forecast_pressure.set_yticklabels([])
        pressure_levels = np.arange(500, 1001, 25)
        pressure_contours = self.ax_forecast_pressure.contour(
            time_grid,
            height_grid,
            pressures,
            levels=pressure_levels,
            colors="black",
            linewidths=0.8,
        )
        self.ax_forecast_pressure.clabel(
            pressure_contours, inline=True, fontsize=7, fmt="%d"
        )
        for axis in (
            self.ax_forecast_temperature,
            self.ax_forecast_wind,
            self.ax_forecast_pressure,
        ):
            axis.xaxis_date()
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
            axis.tick_params(axis="x", rotation=35)
            axis.grid(True, alpha=0.2)
        self.ax_forecast_pressure.set_xlabel("Time (Europe/Warsaw)")
        self.ax_forecast_temperature.set_xlim(time_values[0], time_values[-1])
        self.forecast_figure.tight_layout()
        self.forecast_canvas.draw_idle()
        self.forecast_times = time_values

    def _forecast_plot_clicked(self, event: object) -> None:
        if (
            getattr(event, "inaxes", None)
            not in (
                self.ax_forecast_temperature,
                self.ax_forecast_wind,
                self.ax_forecast_pressure,
            )
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
            or not self.weather_profiles
        ):
            return
        time_index = int(
            np.argmin(np.abs(self.forecast_times - float(event.xdata)))
        )
        height_index = int(
            np.argmin(
                np.abs(self.weather_profiles[0].heights_m - float(event.ydata))
            )
        )
        profile = self.weather_profiles[time_index]
        height = float(profile.heights_m[height_index])
        east = float(profile.wind_east_ms[height_index])
        north = float(profile.wind_north_ms[height_index])
        direction = wind_direction_from_components(east, north)
        speed = math.hypot(east, north)
        timestamp = datetime.fromisoformat(profile.timestamp)
        direction_text = "calm" if direction is None else f"{direction:.1f}°"
        self.forecast_detail.setText(
            f"Selected point: {timestamp:%d.%m.%Y %H:%M} Europe/Warsaw, "
            f"{height:.0f} m above ground — wind {speed:.2f} m/s, direction FROM "
            f"{direction_text}; temperature {profile.temperature_c[height_index]:.1f} °C; "
            f"pressure {profile.pressure_hpa[height_index]:.1f} hPa. "
            f"Open-Meteo model: {self.weather_model.currentText()}."
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
            f"Forecast at {height:.0f} m, {profile.timestamp}: "
            f"T {temperature:.1f} °C (measurement−forecast error "
            f"{self.measured_temperature.value() - temperature:+.1f} °C), "
            f"p {pressure:.1f} hPa "
            f"({self.measured_pressure.value() - pressure:+.1f} hPa), "
            f"wind {forecast_speed:.1f} m/s "
            f"({self.measured_wind_speed.value() - forecast_speed:+.1f} m/s), "
            f"direction FROM {forecast_direction:.0f}° "
            f"(forecast−measurement direction difference {wind_direction_error:+.0f}°)."
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
                f"For Cd={cd:.2f}, m={self.mass.value():.3f} kg and "
                f"v={target_speed:.2f} m/s: A≈{area_for_target * 10000:.0f} cm² "
                f"({area_for_target:.3f} m²), circular canopy diameter≈"
                f"{diameter_for_target * 100:.1f} cm. Speed for the current "
                f"area {self.canopy_area.value():.0f} cm²: "
                f"{speed_for_area:.2f} m/s."
            )
            self.apply_area_button.setEnabled(True)
        else:
            self.design_result.setText(
                f"For the current canopy A={self.canopy_area.value():.0f} cm² "
                f"and Cd={cd:.2f}: terminal speed≈{speed_for_area:.2f} m/s. "
                f"Area for {target_speed:.2f} m/s would require "
                f"{area_for_target * 10000:.0f} cm² (circular diameter "
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
            f"Current thrust: {self.thrust.value():.2f} N, "
            f"time: {'until landing' if self.continuous_thrust.isChecked() else f'{self.thrust_duration.value():.1f} s'}, "
            f"modelled tilt: {maximum_tilt:.1f}°. "
            f"Configured limit: {self.max_tilt.value():.0f}°. "
            "The limit is a user-defined design criterion, not a certification."
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
            QMessageBox.warning(self, "No weather data", "Fetch a forecast or select Manual weather.")
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
                    "The estimated tilt already exceeds the configured limit with the motors off. "
                    "Change the canopy, conditions, or criterion."
                )
                return
            candidate_thrusts = np.arange(0.5, 25.01, 0.5)
            unsafe_thrust = next(
                (float(candidate) for candidate in candidate_thrusts if not safe(float(candidate))),
                None,
            )
            if unsafe_thrust is None:
                self.engine_result.setText(
                    f"No exceedance was found in the scanned range of 0–25 N (in 0.5 N increments) "
                    f"at the configured tilt limit of {self.max_tilt.value():.0f}°. "
                    "This result does not determine a structural limit."
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
                f"Model-based threshold: approximately {low:.2f} N total thrust "
                f"(tilt limit {self.max_tilt.value():.0f}°, "
                f"scan 0–25 N in 0.5 N increments followed by 12 interval subdivisions). "
                "Use a safety margin; "
                "this does not replace static tests and flight trials."
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.thrust_limit_button.setEnabled(True)

    def calculate_correction(self) -> None:
        profile = self._profile()
        if profile is None:
            QMessageBox.warning(self, "No weather data", "Fetch a forecast or select Manual weather.")
            return
        if self.thrust.value() <= 0:
            self.engine_result.setText("Set a positive motor thrust to search for a correction.")
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
                raise ValueError("No correction option was found.")
            self.engine_result.setText(
                f"Best tested continuous burn segment: "
                f"azimuth {best[1]:.0f}°, "
                f"{best[2]:.1f} s at {self.thrust.value():.2f} N; "
                f"predicted miss distance from target {best[0]:.1f} m. "
                "The search tested 24 azimuths at 15° intervals and 9 durations from 0 to the descent time "
                "(max. 180 s), choosing the smallest distance between the landing point "
                "and the target. The current wind profile, mass, canopy, and thrust were used; "
                "this is a model-based suggestion, not active in-flight control."
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
                "No trajectory",
                "First select weather and calculate a valid trajectory.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CanSat trajectory",
            "cansat_trajectory.csv",
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
                        "time_s",
                        "latitude_WGS84_deg",
                        "longitude_WGS84_deg",
                        "altitude_AGL_m",
                        "altitude_ASL_m",
                        "horizontal_speed_m_s",
                        "vertical_speed_m_s",
                        "canopy_tilt_deg",
                        "wind_speed_m_s",
                        "wind_direction_from_deg",
                        "weather_profile",
                        "launch_latitude_deg",
                        "launch_longitude_deg",
                        "mass_kg",
                        "canopy_area_cm2",
                        "drag_coefficient_Cd",
                        "thrust_N",
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
                "Could not save CSV",
                f"File: {path}\nDetails: {error}",
            )
            return
        QMessageBox.information(
            self, "Trajectory saved", f"Saved {len(times)} points to:\n{path}"
        )

    def run_scenarios(self) -> None:
        profile = self._profile()
        if profile is None:
            QMessageBox.warning(self, "No weather data", "Fetch a forecast or select Manual weather.")
            return
        self.scenario_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            heading_rad = math.radians(self.heading.value())
            gust_east = 2 * math.sin(heading_rad)
            gust_north = 2 * math.cos(heading_rad)
            scenarios = [
                {
                    "name": "Motors off",
                    "description": "thrust=0 N; current weather and canopy",
                    "profile": profile,
                    "thrust": 0.0,
                    "duration": 0.0,
                    "tilt_effect": None,
                },
                {
                    "name": "Current configuration",
                    "description": "all current panel settings",
                    "profile": profile,
                    "thrust": None,
                    "duration": None,
                    "tilt_effect": None,
                },
                (
                    "Favourable gust +2 m/s",
                    profile.shifted(gust_east, gust_north),
                    None,
                    None,
                    None,
                    "adds +2 m/s at every altitude in the motor-thrust direction",
                ),
                (
                    "Opposing gust −2 m/s",
                    profile.shifted(-gust_east, -gust_north),
                    None,
                    None,
                    None,
                    "adds −2 m/s at every altitude, opposite to the motor-thrust direction",
                ),
                (
                    "Tilt coupling disabled",
                    profile,
                    None,
                    None,
                    False,
                    "same as the current configuration, but vertical drag does not depend on tilt angle",
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
                label="Deterministic scenarios"
            )
            self.ax_scenarios.scatter(
                [target_east], [target_north], marker="*", color="red",
                s=140, label="Operator target"
            )
            self.ax_scenarios.set_title("Landing points (local coordinates)")
            self.ax_scenarios.set_xlabel("East of launch (m)")
            self.ax_scenarios.set_ylabel("North of launch (m)")
            self.ax_scenarios.grid(True, alpha=0.3)
            self.ax_scenarios.axis("equal")
            self.ax_scenarios.legend()
            self.scenario_canvas.draw_idle()
            self.scenario_table.insertRow(self.scenario_table.rowCount())
            summary_values = (
                f"Monte Carlo: median ({count} trials)",
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
                f"Monte Carlo: {count} repetitions of the same configuration and selected "
                f"weather hour; each trial adds independent offsets to the "
                f"E and N components, constant over altitude, drawn from N(0, {sigma:.1f}²) m/s. "
                f"The random seed is 2027, so results are reproducible. "
                f"Distance to target: median {np.median(monte_carlo_errors):.1f} m, "
                f"95th percentile {np.percentile(monte_carlo_errors, 95):.1f} m. "
                "This is not a probability forecast: it does not model turbulence, "
                "weather changes over time, or sensor errors."
            )
            self.scenario_details.setText(
                f"<b>Shared data:</b> {profile.timestamp}, "
                f"start {self.latitude.value():.5f}° N, "
                f"{self.longitude.value():.5f}° E; release altitude "
                f"{self.release_height.value():.0f} m; mass "
                f"{self.mass.value():.3f} kg; canopy "
                f"{self.canopy_area.value():.0f} cm², Cd "
                f"{self.drag_coefficient.value():.2f}; panel thrust "
                f"{self.thrust.value():.2f} N; integration step 0.15 s."
                f"<br><b>What the variants change:</b> "
                f"{'; '.join(scenario_descriptions)}."
                "<br><b>What the program calculates:</b> for each variant it integrates "
                "motion until ground contact, then calculates velocity, tilt, time, and "
                "the distance between the landing point and the operator target."
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
