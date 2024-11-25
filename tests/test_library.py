from weather_visualization import WeatherDataPlotter
import pandas as pd

data = pd.read_csv(r"E:\projects\Python\PP\src\london_weather.csv")

plotter = WeatherDataPlotter(data, 'date')

plotter.plot_temperature_with_scales(
    "min_temp",
    "mean_temp",
    "max_temp",
)
plotter.plot_snow_depth("snow_depth")
plotter.plot_radiation("global_radiation", "mean_temp")
plotter.plot_precipitation("precipitation")
plotter.plot_cloud_cover("cloud_cover")
plotter.plot_pressure("pressure", "date")
plotter.plot_sunshine("sunshine","date")
plotter.plot_sunshine_year(2005, "sunshine", "date")
plotter.weather_report("date", "sunshine", ["min_temp", "mean_temp", "max_temp"], "precipitation")

'''data1 = pd.read_csv(r"E:\projects\Python\PP\tests\seattle-weather.csv")

plotters = WeatherDataPlotter(data1, "date")

plotters.plot_precipitation("precipitation")'''
