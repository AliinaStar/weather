# Weather Conditions Visualization

A library for visualizing weather conditions that allows you to create graphs of temperature, precipitation, snow cover, wind speed, cloudiness, sunshine hours, and solar radiation.

## Installation

```bash
pip install weather_visualization
```

## Usage
```
from weather_visualization import WeatherDataPlotter

# Loading data
data = pd.read_csv(r"E:\projects\Python\PP\src\london_weather.csv")

# Creating visualizations
plotter = WeatherDataPlotter(data, 'date')

# Available visualizations
```
plotter.plot_temperature_with_scales("min_temp", "mean_temp", "max_temp")    # Temperature
![Temperature](images/temperature.png)    
plotter.plot_snow_depth("snow_depth")                                        # Snow cover
![Snow depth](images/snow.png)    
plotter.plot_radiation("global_radiation", "mean_temp")                      # Solar radiation
![Radiation](images/radiation.png)    
plotter.plot_precipitation("precipitation")                                  # Precipitation
![Precipitation](images/precipitation.png)    
plotter.plot_cloud_cover("cloud_cover")                                      # Cloud cover
![Cloud cover](images/cloud.png)   
plotter.plot_pressure("pressure", "date")                                    # Pressure
![Pressure](images/pressure.png)   
plotter.plot_sunshine("sunshine","date")                                     # Sunshine during decade
![Sunshine decade](images/sun_decade.png)   
plotter.plot_sunshine_year(2005, "sunshine", "date")                         # Sunshine during year
![Sunshine year](images/sun_year.png)   
plotter.weather_report("date", "sunshine", "mean_temp", "precipitation")     # General report about sunny days, prepitiation and temperatute
```