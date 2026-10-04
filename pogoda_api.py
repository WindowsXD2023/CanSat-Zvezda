"""Weather forecast scraper module for CanSat projects."""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List


class WeatherScraper:
    """Fetches vertical profile weather forecasts from Open-Meteo API."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    TIMEZONE = "Europe/Warsaw"
    WIND_UNIT = "ms"
    FORECAST_DAYS = 2
    
    # Pressure levels to fetch (hPa) - from surface up to ~2500m
    PRESSURE_LEVELS = list(range(975, 724, -25))  # 975, 950, ..., 750 hPa

    def __init__(self, latitude: float, longitude: float, 
                 model: Optional[str] = None):
        """Initialize scraper with location and optional model.
        
        Args:
            latitude: Geographic latitude (-90 to 90)
            longitude: Geographic longitude (-180 to 180)
            model: Weather model ('gfs_seamless', None for automatic)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.model = model
        self.raw_data = None
        self.processed_data = None
        self.elevation = None

    def _build_query_params(self) -> Dict:
        """Build API request parameters."""
        # Variables for each pressure level
        variables = []
        for level in self.PRESSURE_LEVELS:
            variables.extend([
                f"temperature_{level}hPa",
                f"wind_speed_{level}hPa",
                f"wind_direction_{level}hPa",
                f"geopotential_height_{level}hPa",
            ])
        
        # Surface-level variables
        variables.extend([
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
        ])
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(variables),
            "wind_speed_unit": self.WIND_UNIT,
            "timezone": self.TIMEZONE,
            "forecast_days": self.FORECAST_DAYS,
        }
        
        if self.model is not None:
            params["models"] = self.model
        
        return params

    def fetch(self, timeout: int = 30) -> Dict:
        """Fetch raw weather data from API.
        
        Returns:
            Dictionary with hourly data, timestamps, and elevation.
            
        Raises:
            requests.RequestException: On network/API errors
            ValueError: If insufficient vertical data available
        """
        params = self._build_query_params()
        
        response = requests.get(self.BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        self.raw_data = data
        self.elevation = data.get("elevation", 0)
        
        # Validate data availability
        hourly = data.get("hourly", {})
        valid_levels = sum(
            any(v is not None for v in hourly.get(f"temperature_{lvl}hPa", []))
            for lvl in self.PRESSURE_LEVELS
        )
        
        if valid_levels < 2:
            raise ValueError(
                f"Weather model '{self.model}' returned insufficient "
                f"vertical data for lat={self.latitude}, lon={self.longitude}"
            )
        
        # Check max altitude coverage
        times = hourly.get("time", [])
        max_heights = {}
        for i, t in enumerate(times):
            max_h = 0
            for lvl in self.PRESSURE_LEVELS:
                h_vals = hourly.get(f"geopotential_height_{lvl}hPa", [])
                if i < len(h_vals) and h_vals[i] is not None:
                    h = h_vals[i] - self.elevation
                    max_h = max(max_h, h)
            max_heights[t] = max_h
        
        if any(h < 2500 for h in max_heights.values()):
            raise ValueError(
                "Selected model does not provide data up to 2500m. "
                "Try using 'gfs_seamless' or automatic model."
            )
        
        return {
            "timestamps": pd.to_datetime(hourly["time"]),
            "elevation": self.elevation,
            "hourly": hourly,
        }

    def interpolate_to_heights(self, target_heights_m: np.ndarray = None) -> Dict:
        """Interpolate weather data to regular height grid.
        
        Args:
            target_heights_m: Array of target heights in meters (default: 0-2500m, 100m steps)
            
        Returns:
            Dictionary with temperature, wind speed/direction, pressure at each height/time.
        """
        if self.raw_data is None:
            raise RuntimeError("Must call fetch() first")
        
        if target_heights_m is None:
            target_heights_m = np.arange(0, 2501, 100)  # 0 to 2500m, every 100m
        
        hourly = self.raw_data["hourly"]
        n_times = len(hourly["time"])
        
        # Build source height arrays
        source_heights = []
        source_heights.append(np.full(n_times, 10.0))  # Surface wind at 10m
        
        for level in self.PRESSURE_LEVELS:
            h_key = f"geopotential_height_{level}hPa"
            heights = np.array(hourly.get(h_key, []), dtype=float) - self.elevation
            source_heights.append(heights)
        source_heights = np.vstack(source_heights)
        
        # Build variable arrays
        temp_src = np.vstack([
            np.array(hourly.get("temperature_2m", []), dtype=float),
            *[np.array(hourly.get(f"temperature_{lvl}hPa", []), dtype=float)
              for lvl in self.PRESSURE_LEVELS]
        ])
        
        wind_speed_src = np.vstack([
            np.array(hourly.get("wind_speed_10m", []), dtype=float),
            *[np.array(hourly.get(f"wind_speed_{lvl}hPa", []), dtype=float)
              for lvl in self.PRESSURE_LEVELS]
        ])
        
        wind_dir_src = np.vstack([
            np.array(hourly.get("wind_direction_10m", []), dtype=float),
            *[np.array(hourly.get(f"wind_direction_{lvl}hPa", []), dtype=float)
              for lvl in self.PRESSURE_LEVELS]
        ])
        
        pressure_src = np.vstack([
            np.array(hourly.get("surface_pressure", []), dtype=float),
            *[np.full(n_times, float(level), dtype=float)
              for level in self.PRESSURE_LEVELS]
        ])
        
        # Convert wind to vector components
        wind_dir_rad = np.deg2rad(wind_dir_src)
        wind_u_src = -wind_speed_src * np.sin(wind_dir_rad)
        wind_v_src = -wind_speed_src * np.cos(wind_dir_rad)
        
        # Interpolate each variable
        def interp_to_heights(src_heights, src_values, target):
            result = np.full((len(target), src_values.shape[1]), np.nan)
            for i in range(src_values.shape[1]):
                valid = np.isfinite(src_heights[:, i]) & np.isfinite(src_values[:, i])
                if np.sum(valid) >= 2:
                    idx = np.argsort(src_heights[valid, i])
                    result[:, i] = np.interp(
                        target,
                        src_heights[valid, i][idx],
                        src_values[valid, i][idx]
                    )
            return result
        
        temp_interp = interp_to_heights(source_heights, temp_src, target_heights_m)
        wind_u_interp = interp_to_heights(source_heights, wind_u_src, target_heights_m)
        wind_v_interp = interp_to_heights(source_heights, wind_v_src, target_heights_m)
        pressure_interp = interp_to_heights(source_heights, pressure_src, target_heights_m)
        
        # Reconstruct wind from components
        wind_speed = np.hypot(wind_u_interp, wind_v_interp)
        wind_dir = np.mod(np.rad2deg(np.arctan2(-wind_u_interp, -wind_v_interp)), 360)
        
        # Build output dictionary
        processed = {"timestamp": pd.to_datetime(self.raw_data["hourly"]["time"])}
        processed["elevation_m"] = self.elevation
        processed["target_heights_m"] = target_heights_m
        
        for i, h in enumerate(target_heights_m):
            processed[f"temperature_{h}m"] = temp_interp[i]
            processed[f"wind_speed_{h}m"] = wind_speed[i]
            processed[f"wind_direction_{h}m"] = wind_dir[i]
            processed[f"pressure_{h}m"] = pressure_interp[i]
        
        self.processed_data = processed
        return processed

    def get_height_profiles(self, height_m: int = 500) -> pd.DataFrame:
        """Get single-time snapshot at specific height for all times.
        
        Args:
            height_m: Desired height in meters (closest will be used)
            
        Returns:
            DataFrame with time index and variables at selected height.
        """
        if self.processed_data is None:
            self.interpolate_to_heights()
        
        # Find closest available height
        available_heights = self.processed_data["target_heights_m"]
        closest_idx = np.argmin(np.abs(available_heights - height_m))
        closest_h = available_heights[closest_idx]
        
        df = pd.DataFrame({
            "timestamp": self.processed_data["timestamp"],
            "temperature_C": self.processed_data[f"temperature_{closest_h}m"],
            "wind_speed_ms": self.processed_data[f"wind_speed_{closest_h}m"],
            "wind_direction_deg": self.processed_data[f"wind_direction_{closest_h}m"],
            "pressure_hPa": self.processed_data[f"pressure_{closest_h}m"],
        })
        df.set_index("timestamp", inplace=True)
        return df

    def get_wind_profile_at_time(self, timestamp_str: str = None,
                                height_max: int = 2500) -> pd.DataFrame:
        """Get vertical wind profile at specific time."""
        if self.processed_data is None:
            self.interpolate_to_heights()
        
        if timestamp_str is None:
            ts = self.processed_data["timestamp"][0]
        else:
            ts = pd.to_datetime(timestamp_str)
            if ts not in self.processed_data["timestamp"].values:  # FIX: .values added
                raise ValueError(f"Timestamp '{ts}' not in available times")
        
        # Find index of this timestamp
        col_idx = np.where(self.processed_data["timestamp"] == ts)[0][0]  # FIX: Use np.where
            
        # Filter heights up to max
        valid_heights = self.processed_data["target_heights_m"]
        valid_heights = valid_heights[valid_heights <= height_max]
        
        df_data = {"height_m": valid_heights}
        for h in valid_heights:
            df_data[f"wind_speed_{h}m"] = self.processed_data[f"wind_speed_{h}m"][col_idx]
            df_data[f"wind_direction_{h}m"] = self.processed_data[f"wind_direction_{h}m"][col_idx]
        
        df = pd.DataFrame(df_data).set_index("height_m")
        return df


# === Convenience Functions ===

def fetch_weather(latitude: float, longitude: float, 
                  model: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
    """Quick function to fetch and interpolate weather data.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        model: Optional weather model name
        
    Returns:
        Tuple of (target_heights, processed_data_dict)
    """
    scraper = WeatherScraper(latitude, longitude, model)
    scraper.fetch()
    processed = scraper.interpolate_to_heights()
    return scraper.processed_data["target_heights_m"], processed


def get_surface_conditions(latitude: float, longitude: float,
                           hours_ahead: int = 0) -> Dict:
    """Get surface-level forecast at specific hour offset.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        hours_ahead: Hours from now (default: 0 = current)
        
    Returns:
        Dictionary with temperature, wind, pressure at surface.
    """
    scraper = WeatherScraper(latitude, longitude)
    data = scraper.fetch()
    hourly = data["hourly"]
    
    # Handle hours offset (clamp to available data)
    n_hours = len(hourly["time"])
    idx = min(hours_ahead, n_hours - 1)
    
    return {
        "temperature_C": hourly["temperature_2m"][idx],
        "wind_speed_ms": hourly["wind_speed_10m"][idx],
        "wind_direction_deg": hourly["wind_direction_10m"][idx],
        "pressure_hPa": hourly["surface_pressure"][idx],
        "timestamp": data["timestamps"][idx],
        "elevation_m": scraper.elevation,
    }


if __name__ == "__main__":
    # Basic usage
    print("=" * 50)
    print("CanSat Weather Scraper Demo")
    print("=" * 50)
    
    # Fetch data for Kraków (approximate CanSat launch site)
    lat, lon = 50.0647, 19.9450  # Krakow, Poland
    
    scraper = WeatherScraper(lat, lon, model="gfs_seamless")
    
    try:
        print(f"\nFetching weather for {lat}, {lon}...")
        scraper.fetch()
        print(f"Elevation: {scraper.elevation:.1f} m")
        
        # FIX: interpolate_to_heights() returns dict only, access heights from it
        processed = scraper.interpolate_to_heights()
        heights = processed["target_heights_m"]
        print(f"Available heights: {heights.min():.0f}m - {heights.max():.0f}m")
        
        # Show surface conditions for next 6 hours
        print("\n--- Surface Forecast ---")
        for h in range(6):
            cond = get_surface_conditions(lat, lon, hours_ahead=h)
            print(f"Hour {h}: {cond['temperature_C']:.1f}°C, "
                  f"Wind {cond['wind_speed_ms']:.1f} m/s @ {cond['wind_direction_deg']:.0f}°")
        
        # Show vertical profile at specific time (FIXED: use timestamp directly)
        print("\n--- Wind Profile at First Timepoint ---")
        first_ts = processed["timestamp"][0]
        profile = scraper.get_wind_profile_at_time(timestamp_str=str(first_ts))
        print(profile.to_string())
        
        # Export to CSV for further use
        scraper.get_height_profiles(500).to_csv("weather_profile_500m.csv")
        print("\nExported 500m profile to 'weather_profile_500m.csv'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()