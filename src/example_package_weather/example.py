import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
class WeatherDataFetcher:
    def _fetch_weather_data(self, source: str) -> pd.DataFrame:
        try:
            weather_data = pd.read_csv(source)

            required_columns = [
                "date", "cloud_cover", "sunshine", "global_radiation",
                "max_temp", "mean_temp", "min_temp", "precipitation",
                "pressure", "snow_depth"
            ]
            missing_columns = [col for col in required_columns if col not in weather_data.columns]

            if missing_columns:
                raise ValueError(f"Відсутні необхідні колонки: {', '.join(missing_columns)}")

            return weather_data
        except Exception as e:
            print(f"Помилка: {e}")
        return pd.DataFrame()

class WeatherDataPlotter:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_temperature_with_scales(self):
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")


            self.data['date'] = pd.to_datetime(self.data['date'], format='%Y%m%d', errors='coerce')
            self.data['decade'] = (self.data['date'].dt.year // 10) * 10


            grouped = self.data.groupby('decade').agg({
                'min_temp': 'mean',
                'mean_temp': 'mean',
                'max_temp': 'mean'
            }).reset_index()


            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(grouped['decade']))
            bar_width = 0.3


            colors = {
                'min_temp': 'blue',
                'mean_temp': 'yellow',
                'max_temp': 'red'
            }

            for i, decade in enumerate(x):
                for temp_type, color in colors.items():
                    temp_value = grouped[temp_type][i]
                    ax.bar(
                        decade, temp_value, width=bar_width, color=color, alpha=0.7, edgecolor='black'
                    )


                    scale_y = np.linspace(0, temp_value, 5)  # 5 позначок
                    for y in scale_y:
                        ax.plot(
                            [decade - bar_width / 4, decade + bar_width / 4],  # маленькі горизонтальні штрихи
                            [y, y],
                            color='black',
                            linewidth=0.5
                        )


                    ax.scatter(
                        [decade], [0], color=color, s=200, edgecolor='black', zorder=3
                    )


            ax.set_xticks(x)
            ax.set_xticklabels(grouped['decade'])
            ax.set_xlabel("Десятиліття")
            ax.set_ylabel("Температура (°C)")
            ax.set_title("Температури по десятиліттях у вигляді термометрів зі шкалами")
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Помилка: {e}")


fetcher = WeatherDataFetcher()
df = fetcher._fetch_weather_data(r"D:\weather\src\london_weather.csv")

if not df.empty:
    plotter = WeatherDataPlotter(df)
    plotter.plot_temperature_with_scales()
