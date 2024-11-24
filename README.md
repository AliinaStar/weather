# Weather Conditions Visualization

A library for visualizing weather conditions that allows you to create graphs of temperature, precipitation, snow cover, wind speed, cloudiness, sunshine hours, and solar radiation.

## Installation

```bash
pip install weather_visualization
```

## Usage
```
from weather_visualization import WeatherDataFetcher, WeatherDataPlotter

# Loading data
fetcher = WeatherDataFetcher()
data = fetcher._fetch_weather_data("your_data.csv")

# Creating visualizations
plotter = WeatherDataPlotter(data)

# Available visualizations
plotter.plot_temperature_with_scales()  # Temperature by decades
plotter.plot_snow_depth()               # Snow cover
plotter.plot_radiation()                # Solar radiation
plotter.plot_precipitation()            # Precipitation
plotter.plot_cloud_cover()              # Cloud cover
plotter.plot_pressure()                 # Pressure
plotter.plot_sunshine()                 # Sunshine during decade
plotter.plot_sunshine_year()            # Sunshine during year
plotter.weather_report()                # General report about sunny days, prepitiation and temperatute
```