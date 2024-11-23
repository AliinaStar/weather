import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    def plot_radiation(self):
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
            
            grouped = self.data.groupby('decade')
            decades = list(grouped.groups.keys())
            num_decades = len(decades)
            cols = 3
            rows = (num_decades + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            axes = axes.flatten()
            
            for idx, (decade, group) in enumerate(grouped):
                axes[idx].scatter(
                    x=group['global_radiation'],
                    y=group['mean_temp'],
                    alpha=0.5,
                    color='orange'
                )
                
                axes[idx].set_title(f"{decade}s")
                axes[idx].set_xlabel("Глобальна радіація")
                axes[idx].set_ylabel("Середня температура")
                axes[idx].grid(True, linestyle='--', alpha=0.7)
            
            for idx in range(num_decades, len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Помилка: {e}")

    def plot_precipitation(self):
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
            
            grouped_data = self.data.groupby('decade')['precipitation'].mean().reset_index()
        
            fig, ax = plt.subplots(figsize=(12, 6))

            x = grouped_data['decade']
            y = grouped_data['precipitation']
            max_value = max(y)

            for i, (x_val, y_val) in enumerate(zip(x, y)):
                ax.bar(
                    x=x_val,
                    height=max_value*1.3,
                    width=1.5,
                    color='none',
                    edgecolor='Black',
                    linewidth=1.0,
                    zorder=4
                )
                
                ax.bar(
                    x=x_val,
                    height=y_val,  
                    width=1.5,
                    color='lightblue',
                    edgecolor='none',
                    zorder=3
                )

            ax.set_title("Середня кількість опадів по десятиліттях")
            ax.set_xlabel("Десятиліття")
            ax.set_ylabel("Середня кількість опадів")
            ax.grid(True, which='major', linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)

            plt.show()
        except Exception as e:
            print(f"Помилка: {e}")
    
    def cloud_cover(self):
        """
        Візуалізує покриття хмарами для кожного десятиліття у вигляді хмарок з точок.
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
            
            if 'cloud_cover' not in self.data.columns or self.data['cloud_cover'].dropna().empty:
                raise ValueError("Немає даних про покриття хмарами.")

            grouped_data = self.data.groupby('decade')['cloud_cover'].apply(list)
            
            decades = grouped_data.index
            num_decades = len(decades)
            cols = 3  
            rows = (num_decades + cols - 1) // cols  

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
            axes = axes.flatten() 

            for idx, (decade, cloud_values) in enumerate(grouped_data.items()):

                np.random.seed(42 + idx) 
                x = np.random.normal(loc=50, scale=15, size=len(cloud_values))
                y = np.random.normal(loc=50, scale=10, size=len(cloud_values))
                sizes = np.array(cloud_values) * 5


                axes[idx].scatter(x, y, s=sizes, alpha=0.5, color='lightblue', edgecolor='blue' )
                axes[idx].set_title(f"{decade}s", fontsize=14)
                axes[idx].set_xlim(0, 100)
                axes[idx].set_ylim(0, 100)
                axes[idx].set_xlabel("Ширина розташування(придумати шось нормальне)", fontsize=12)
                axes[idx].set_ylabel("Висота розташування(придумати шось нормальне)", fontsize=12)
                axes[idx].grid(True, linestyle='--', alpha=0.7)


            for idx in range(num_decades, len(axes)):
                fig.delaxes(axes[idx])

            fig.suptitle("Покриття хмарами для кожного десятиліття", fontsize=16, y=1.02)
            plt.show()

        except Exception as e:
            print(f"Помилка: {e}")



fetcher = WeatherDataFetcher()
df = fetcher._fetch_weather_data(r"E:\projects\Python\PP\src\london_weather.csv")

if not df.empty:
    plotter = WeatherDataPlotter(df)
    plotter.plot_temperature_with_scales()
    plotter.plot_snow_depth_by_decade()
    plotter.plot_radiation()
    plotter.plot_precipitation()
    plotter.cloud_cover()
