import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from termcolor import colored
from tabulate import tabulate
from scipy.signal import savgol_filter


class WeatherDataPlotter:
    def __init__(self, data: pd.DataFrame, date_col: str):
        """
        Ініціалізує клас WeatherDataPlotter.

        :param data: DataFrame з погодними даними.
        :param date_col: Назва колонки з датами.
        """
        self.data = data.copy()
        self.date_col = date_col

        # Перевірка наявності колонки
        if date_col not in self.data.columns:
            raise ValueError(f"Колонка з датами '{date_col}' відсутня у датасеті.")

        # Спроба визначити формат дати
        try:
            # Якщо дати вже у форматі YYYYMMDD
            if pd.to_datetime(self.data[date_col], format='%Y%m%d', errors='coerce').notna().all():
                self.data[date_col] = pd.to_datetime(self.data[date_col], format='%Y%m%d')
            # Якщо дати у форматі YYYY-MM-DD
            elif pd.to_datetime(self.data[date_col], format='%Y-%m-%d', errors='coerce').notna().all():
                self.data[date_col] = pd.to_datetime(self.data[date_col], format='%Y-%m-%d')
            else:
                raise ValueError("Формат дати у колонці невідомий.")
        except Exception as e:
            raise ValueError(f"Помилка під час конвертації дат: {e}")

        # Додавання десятиліття
        self.data['decade'] = (self.data[date_col].dt.year // 10) * 10

        # Перетворення дат у формат YYYYMMDD
        self.data[date_col] = self.data[date_col].dt.strftime('%Y%m%d')



    def plot_temperature_with_scales(self, min_col, mean_col, max_col):
        """
        Creates a specialized temperature visualization with thermometer-like scales by decades.

        Plots min, mean, and max temperatures as bars with measurement scales and thermometer-style indicators.
        Handles empty dataset cases and general plotting errors.

        :return: None
        :raises ValueError: If the dataset is empty
        """
        try:

            if any(col not in self.data.columns for col in [min_col, mean_col, max_col]):
                raise ValueError(f"Однієї або декількох необхідних колонок немає: {min_col}, {mean_col}, {max_col}")

            grouped = self.data.groupby('decade').agg({
                min_col: 'mean',
                mean_col: 'mean',
                max_col: 'mean'
            }).reset_index()


            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(grouped['decade']))
            bar_width = 0.3


            colors = {
                min_col: 'blue',
                mean_col: 'yellow',
                max_col: 'red'
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

    def plot_snow_depth(self, snow_depth):
        """
        Creates density plots of snow depth distribution for each decade.

        Generates a grid of KDE plots showing snow depth patterns across decades.
        Handles empty datasets and missing snow depth data.

        :return: None
        :raises ValueError: If the dataset is empty or snow depth data is missing
        """
        try:

            if snow_depth not in self.data.columns or self.data[snow_depth].dropna().empty:
                raise ValueError(f"Колонка '{snow_depth}' відсутня або пуста.")


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

    def plot_radiation(self, radiation_col, temp_col):
        """
        Creates scatter plots with regression lines showing relationship between global radiation and mean temperature by decades.

        Plots temperature vs radiation data points and trend lines for each decade in a grid layout.
        Handles empty dataset cases.

        :return: None
        :raises ValueError: If the dataset is empty
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
            
            if any(col not in self.data.columns for col in [radiation_col, temp_col]):
                raise ValueError(f"Однієї або декількох колонок немає: {radiation_col}, {temp_col}")

            grouped = self.data.groupby('decade')
            decades = list(grouped.groups.keys())
            num_decades = len(decades)
            cols = 3
            rows = (num_decades + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            axes = axes.flatten()

            for idx, (decade, group) in enumerate(grouped):
                axes[idx].scatter(
                    x=group[radiation_col],
                    y=group[temp_col],
                    alpha=0.5,
                    color='lightgreen'
                )

                sns.regplot(
                    data=group,
                    x=group[radiation_col],
                    y=group[temp_col],
                    ax=axes[idx], 
                    color='green',
                    scatter=False
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

    def plot_precipitation(self, precipitation_col):
        """
            Visualizes average precipitation by decades using special bar chart with empty frame containers.
            
            Creates a bar chart where each decade's precipitation is shown within a standardized frame.
            Handles empty dataset cases.

            :return: None
            :raises ValueError: If the dataset is empty
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
            
            if precipitation_col not in self.data.columns not in self.data.columns:
                raise ValueError(f"Однієї або обох колонок немає: {precipitation_col}")
            
            grouped_data = self.data.groupby('decade')[precipitation_col].mean().reset_index()

            fig, ax = plt.subplots(figsize=(12, 6))
            x = grouped_data['decade']
            y = grouped_data[precipitation_col]
            max_value = max(y)
            for i, (x_val, y_val) in enumerate(zip(x, y)):
                ax.bar(
                    x=x_val,
                    height=max_value * 1.3,
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
     
     
    def plot_cloud_cover(self, cloud_cover_col):
        """
            Visualizes cloud cover for each decade using dot-based cloud representations.
            
            Creates a grid of scatter plots where point sizes represent cloud cover values.
            Handles empty dataset and missing cloud cover data cases.

            :return: None
            :raises ValueError: If the dataset is empty or cloud cover data is missing
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
            
            if cloud_cover_col not in self.data.columns not in self.data.columns:
                raise ValueError(f"Однієї або обох колонок немає: {cloud_cover_col}")

            grouped_data = self.data.groupby('decade')[cloud_cover_col].apply(list)
            
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

    def plot_sunshine(self, sunshine_col, date_col):
        """
        Visualizes sunshine hours for each decade using sun-shaped radial plots.
        
        Each year is represented by a ray, where length and color intensity indicate sunshine amount.
        Creates a grid of visualizations with color scale indicator.

        :return: None
        :raises ValueError: If the dataset is empty or required columns (decade, sunshine, date) are missing
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
    
            if any(col not in self.data.columns for col in [sunshine_col, date_col]):
                raise ValueError(f"Однієї або обох колонок немає: {sunshine_col}, {date_col}")
    
            self.data['year'] = self.data[date_col].dt.year
            grouped_data = self.data.groupby(['decade', 'year'])[sunshine_col].sum().reset_index()
    
            decades = grouped_data['decade'].unique()
    
            cols = 3 
            rows = (len(decades) + cols - 1) // cols 
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
        
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
            for idx, decade in enumerate(decades):
                decade_data = grouped_data[grouped_data['decade'] == decade]
                years = decade_data['year']
                sunshine_values = decade_data[sunshine_col]
    
                norm = plt.Normalize(vmin=sunshine_values.min(), vmax=sunshine_values.max())
                cmap = plt.cm.YlOrRd  
                colors = cmap(norm(sunshine_values))
    
                ax = axes[idx]
                ax.set_aspect('equal')
                num_years = len(years)
                angles = np.linspace(0, 2 * np.pi, num_years, endpoint=False)  
    
                for angle, sun_value, year, color in zip(angles, sunshine_values, years, colors):
                    x = np.cos(angle) * sun_value  
                    y = np.sin(angle) * sun_value  
    
                    ax.plot([0, x], [0, y], color=color, lw=2, alpha=0.8)
    
                    ax.text(x * 1.1, y * 1.1, str(year), fontsize=8, ha='center', va='center')
    
                ax.scatter(0, 0, s=500, color='yellow', zorder=4)
    
                ax.set_xlim(-sunshine_values.max() * 1.2, sunshine_values.max() * 1.2)
                ax.set_ylim(-sunshine_values.max() * 1.2, sunshine_values.max() * 1.2)
                ax.set_title(f"{decade}s", fontsize=12)
                ax.axis('off') 
    
            for idx in range(len(decades), len(axes)):
                fig.delaxes(axes[idx])

            
            fig.suptitle("Сонячна активність за десятиліттями", fontsize=16, y=1.02)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.05, label="Сонячні години")
    
            plt.show()
        except Exception as e:
            print(f"Помилка: {e}")

    def plot_sunshine_year(self, year, sunshine_col, date_col):
        """
        Visualizes sunshine hours for each decade in a sun-shaped pattern.
        Each ray represents one year, with length and color determined by amount of sunshine.

        Creates radial plots where rays length and color intensity show yearly sunshine values.
        Handles empty dataset and missing column cases.

        :return: None
        :raises ValueError: If the dataset is empty or required columns are missing
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")

            if any(col not in self.data.columns for col in [sunshine_col, date_col]):
                raise ValueError(f"Однієї або обох колонок немає: {sunshine_col}, {date_col}")

            self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
            self.data['year'] = self.data['date'].dt.year
            year_data = self.data[self.data['year'] == year]

            if year_data.empty:
                raise ValueError(f"Немає даних для року {year}.")

            year_data = year_data.sort_values(by='date')
            days = year_data['date'].dt.day_of_year
            sunshine_values = year_data[sunshine_col]

            norm = plt.Normalize(vmin=sunshine_values.min(), vmax=sunshine_values.max())
            cmap = plt.cm.YlOrRd  # Жовто-червоний градієнт
            colors = cmap(norm(sunshine_values))

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')

            num_days = len(days)
            angles = np.linspace(0, 2 * np.pi, num_days, endpoint=False)  # Кути для променів

            for angle, sun_value, day_of_year, color in zip(angles, sunshine_values, days, colors):
                x = np.cos(angle) * sun_value  # X-координата кінця променя
                y = np.sin(angle) * sun_value  # Y-координата кінця променя


                ax.plot([0, x], [0, y], color=color, lw=0.7, alpha=0.8)

                if day_of_year % 30 == 0:  
                    ax.text(
                        x * 1.1, y * 1.1,
                        str(day_of_year), fontsize=6, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", edgecolor="gray", alpha=0.8)
                    )

            ax.scatter(0, 0, s=500, color='gold', zorder=3)

            ax.set_xlim(-sunshine_values.max() * 1.2, sunshine_values.max() * 1.2)
            ax.set_ylim(-sunshine_values.max() * 1.2, sunshine_values.max() * 1.2)
            ax.set_title(f"Сонячна активність за {year} рік", fontsize=12, fontweight='regular')
            ax.axis('off')  # Прибираємо осі

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.05, label="Сонячні години")

            plt.show()
        except Exception as e:
            print(f"Помилка: {e}")

    def weather_report(self, date_col, sunshine_col, temp_cols, precipitation_col):
        """
        Generates a comprehensive text-based weather report.

        Creates a formatted report including total records, date range, sunshine hours,
        temperature metrics, and precipitation data.

        :return: None
        :raises ValueError: If the dataset is empty
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
            
            if any(col not in self.data.columns for col in [date_col, sunshine_col, precipitation_col] + temp_cols):
                raise ValueError("Однієї або кількох необхідних колонок немає у датасеті.")

            self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
        
            total_records = len(self.data)
            date_range = (self.data['date'].min(), self.data['date'].max())
            print(colored("\nЗагальна інформація:", "green"))
            print(f"- Кількість записів: {total_records}")
            print(f"- Період даних: {date_range[0].date()} — {date_range[1].date()}")
            
            avg_sunshine = self.data['sunshine'].mean()
            max_sunshine = self.data['sunshine'].max()
            sunniest_day = self.data.loc[self.data['sunshine'].idxmax(), 'date']
    
            print(colored(f"\nСередній сонячний світловий день: {avg_sunshine:.2f} годин/день"))
            print(f"Найсонячніший день: {sunniest_day.date()} ({max_sunshine:.2f} годин)\n")
    
            mean_temp = self.data[temp_cols[1]].mean()
            max_temp = self.data[temp_cols[2]].max()
            min_temp = self.data[temp_cols[0]].min()
            hottest_day = self.data.loc[self.data[temp_cols[2]].idxmax(), date_col]
            coldest_day = self.data.loc[self.data[temp_cols[0]].idxmin(), date_col]
            
            temp_data = [
                ["Середня температура (°C)", f"{mean_temp:.2f}"],
                ["Максимальна температура (°C)", f"{max_temp:.2f} (Дата: {hottest_day.date()})"],
                ["Мінімальна температура (°C)", f"{min_temp:.2f} (Дата: {coldest_day.date()})"]
            ]
            print(colored("Температура:", "blue"))
            print(tabulate(temp_data, headers=["Метрика", "Значення"], tablefmt="fancy_grid"))
            print()
    
            total_precipitation = self.data[precipitation_col].sum()
            avg_precipitation = self.data[precipitation_col].mean()
            max_precipitation_day = self.data.loc[self.data[precipitation_col].idxmax(), date_col]

            precipitation_data = [
                ["Загальна кількість опадів (мм)", f"{total_precipitation:.2f}"],
                ["Середньодобова кількість опадів (мм)", f"{avg_precipitation:.2f}"],
                ["Найбільше опадів (мм)", f"{max_precipitation_day.date()}"]
            ]
            print(colored("Опади:", "blue"))
            print(tabulate(precipitation_data, headers=["Метрика", "Значення"], tablefmt="fancy_grid"))
            print()
            
        except Exception as e:
            print(colored(f"Помилка при створенні звіту: {e}", "red"))

    def plot_pressure(self, pressure_col, date_col):
        """
        Plots the average pressure over time for each decade.

        This method groups the dataset by decades and dates, calculates the mean pressure, 
        and visualizes it in a grid of subplots. Polynomial fitting 
        is applied for smoothing if sufficient data points are available.

        Raises:
            ValueError: If the dataset is empty.
        """
        try:
            if self.data.empty:
                raise ValueError("Датасет порожній.")
            
            if any(col not in self.data.columns for col in [pressure_col]):
                raise ValueError(f"Однієї або обох колонок немає: {pressure_col}")

            grouped_data = self.data.groupby(['decade'])[pressure_col].mean().reset_index()
            decades = grouped_data['decade'].unique()
            num_decades = len(decades)

            cols = 3
            rows = (num_decades + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
            axes = axes.flatten()

            for i, decade in enumerate(decades):
                decade_data = grouped_data[grouped_data['decade'] == decade]

                ax = axes[i]
                y_pressure = decade_data[pressure_col].to_numpy()
                dates = decade_data[date_col]

                ax.plot(
                    dates,
                    y_pressure,
                    color='lightblue',
                    linewidth=2,
                )

                if len(y_pressure) > 5:
                    p = np.polyfit(range(len(dates)), y_pressure, deg=5)
                    y_poly = np.polyval(p, range(len(dates)))
                    ax.plot(
                        dates,
                        y_poly,
                        color='blue',
                    )

                ax.set_title(f"Середній тиск: {decade}s", fontsize=14)
                ax.set_xlabel("Дата")
                ax.set_ylabel("Середній тиск")
                ax.grid(True, linestyle='--', alpha=0.5)

            fig.suptitle("Середній тиск по датах для кожного десятиліття", fontsize=16)

            plt.show()

        except Exception as e:
            print(f"Помилка: {e}")


