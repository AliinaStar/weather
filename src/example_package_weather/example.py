import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class WeatherDataFetcher:
    def __init__(self, required_columns: list[str] = None, date_column: str = "date", date_format: str = "%Y%m%d"):
        """
        Ініціалізує клас DataFetcher.

        :param required_columns: Список необхідних колонок (може бути None, якщо валідація не потрібна).
        :param date_column: Назва колонки з датою (якщо є).
        :param date_format: Формат дати для перевірки (якщо є колонка з датою).
        """
        self.required_columns = required_columns or []
        self.date_column = date_column
        self.date_format = date_format
    def _fetch_weather_data(self, source: str) -> pd.DataFrame:
        """
               Завантажує дані з файлу та виконує перевірки.

               :param source: Шлях до CSV-файлу.
               :return: DataFrame з даними.
               """
        try:
            data = pd.read_csv(source)


            if self.required_columns:
                missing_columns = [col for col in self.required_columns if col not in data.columns]
                if missing_columns:
                    raise ValueError(f"Відсутні необхідні колонки: {', '.join(missing_columns)}")


            if self.date_column in data.columns:
                data[self.date_column] = pd.to_datetime(data[self.date_column], format=self.date_format,
                                                        errors='coerce')
                if data[self.date_column].isna().any():
                    raise ValueError(f"Некоректний формат дати в колонці '{self.date_column}'.")
            return data
        except Exception as e:
            print(f"Помилка: {e}")
        return pd.DataFrame()

class WeatherDataPlotter:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._prepare_data()

    def _prepare_data(self):
        """
        Готує дані, додаючи колонку 'decade'.
        """
        try:
            self.data['date'] = pd.to_datetime(self.data['date'], format='%Y%m%d', errors='coerce')
            self.data['decade'] = (self.data['date'].dt.year // 10) * 10
        except Exception as e:
            print(f"Помилка під час підготовки даних: {e}")

    def plot_temperature_with_scales(self):
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")


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

    def plot_snow_depth_by_decade(self):
        """
        Побудова окремих графіків висоти снігу для кожного десятиліття.
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")



            if 'snow_depth' not in self.data.columns or self.data['snow_depth'].dropna().empty:
                raise ValueError("Немає даних про висоту снігу.")


            grouped_data = self.data.dropna(subset=['snow_depth']).groupby('decade')


            decades = list(grouped_data.groups.keys())
            num_decades = len(decades)
            cols = 3
            rows = (num_decades + cols - 1) // cols


            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
            axes = axes.flatten()


            for idx, (decade, group) in enumerate(grouped_data):
                sns.kdeplot(
                    group['snow_depth'],
                    fill=True,
                    ax=axes[idx],
                    alpha=0.5,
                    color='skyblue'
                )
                axes[idx].set_title(f"{decade}s", fontsize=14)
                axes[idx].set_xlabel("Висота снігу (см)", fontsize=12)
                axes[idx].set_ylabel("Щільність", fontsize=12)
                axes[idx].grid(axis='y', linestyle='--', alpha=0.7)


            for idx in range(num_decades, len(axes)):
                fig.delaxes(axes[idx])


            fig.suptitle("Висота снігу по десятиліттях: Сугроби (KDE)", fontsize=16)
            plt.show()

        except Exception as e:
            print(f"Помилка: {e}")


required_columns = [
    "date", "cloud_cover", "sunshine", "global_radiation",
    "max_temp", "mean_temp", "min_temp", "precipitation",
    "pressure", "snow_depth" 
]
fetcher = WeatherDataFetcher(required_columns=required_columns, date_column="date", date_format="%Y%m%d")
df = fetcher._fetch_weather_data(r"D:\weather\src\london_weather.csv")

if not df.empty:
    plotter = WeatherDataPlotter(df)
    print("Дані успішно завантажені!")
    plotter.plot_temperature_with_scales()
    plotter.plot_snow_depth_by_decade()
