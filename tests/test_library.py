from weather_visualization import WeatherDataFetcher, WeatherDataPlotter

fetcher = WeatherDataFetcher()
data = fetcher._fetch_weather_data(r"E:\projects\Python\PP\tests\test_library.py")

plotter = WeatherDataPlotter(data)

plotter.plot_temperature_with_scales()  
plotter.plot_precipitation()          
