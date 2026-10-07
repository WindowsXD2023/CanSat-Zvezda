import unittest
from unittest.mock import patch

import numpy as np

import symulator_cansat as simulator


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class SimulatorTests(unittest.TestCase):
    def test_parachute_sizing_round_trip(self):
        mass = 0.35
        coefficient = 1.3
        target_speed = 5.0
        density = 1.2
        area = simulator.parachute_area_for_speed(
            mass, coefficient, target_speed, density
        )
        self.assertAlmostEqual(
            simulator.terminal_descent_speed(mass, area, coefficient, density),
            target_speed,
        )
        self.assertGreater(area, 0)

    def test_wind_from_west_moves_landing_east(self):
        weather = simulator.manual_weather_profile(5, 270, 15, 1000)
        flight = simulator.simulate_descent(
            0.35, 0.11, 1.3, 100, 0, 0, 90, weather
        )
        self.assertTrue(flight["landed"])
        self.assertGreater(flight["east"][-1], 0)
        self.assertAlmostEqual(flight["north"][-1], 0, delta=0.1)

    def test_geographic_coordinates_use_launch_location_as_origin(self):
        east, north = simulator.geographic_to_local(50.341, 19.512, 50.34, 19.51)
        self.assertGreater(east, 0)
        self.assertGreater(north, 0)
        origin = simulator.geographic_to_local(50.34, 19.51, 50.34, 19.51)
        self.assertEqual(origin, (0, 0))

    def test_open_meteo_forecast_is_converted_to_wind_components(self):
        times = ["2026-10-07T12:00", "2026-10-07T13:00"]
        hourly = {
            "time": times,
            "temperature_2m": [15, 16],
            "wind_speed_10m": [5, 6],
            "wind_direction_10m": [270, 270],
            "surface_pressure": [970, 969],
        }
        for level in simulator.PRESSURE_LEVELS:
            height = 200 + (975 - level) * 6
            hourly[f"temperature_{level}hPa"] = [15 - (975 - level) * 0.006, 16]
            hourly[f"wind_speed_{level}hPa"] = [5, 6]
            hourly[f"wind_direction_{level}hPa"] = [270, 270]
            hourly[f"geopotential_height_{level}hPa"] = [height, height]

        with patch.object(
            simulator.requests,
            "get",
            return_value=FakeResponse({"elevation": 300, "hourly": hourly}),
        ) as get:
            profiles = simulator.fetch_open_meteo_profiles(50.34, 19.51, None)

        self.assertEqual(len(profiles), 2)
        east, north, _, _, elevation = profiles[0].at(10)
        self.assertAlmostEqual(east, 5, delta=0.01)
        self.assertAlmostEqual(north, 0, delta=0.01)
        self.assertEqual(elevation, 300)
        self.assertEqual(get.call_args.kwargs["timeout"], 20)


if __name__ == "__main__":
    unittest.main()
