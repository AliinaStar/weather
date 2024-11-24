# Weather Conditions Visualization

A library for visualizing weather conditions that allows you to create graphs of temperature, precipitation, snow cover, wind speed, cloudiness, sunshine hours, and solar radiation.

## Installation

```bash
pip install weather-visualization
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
plotter.plot_snow_depth_by_decade()     # Snow cover
plotter.plot_radiation()                # Solar radiation
plotter.plot_precipitation()            # Precipitation
plotter.cloud_cover()                   # Cloud cover
plotter.pressure_plot()                 # Pressure
plotter.sunshine()                      # Sunshine during decade
plotter.sunshine_year()                 # Sunshine during year
plotter.weather_report()                # General report about sunny days, prepitiation and temperatute
```