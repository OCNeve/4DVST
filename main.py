import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset.csv", encoding="utf-8", sep=";")

def make_plot(year):
    print("Selected Year:", year)
    sns.set(style='whitegrid')
    df = pd.read_csv("dataset.csv", encoding="utf-8", sep=";")
    
    # Filter data based on the selected year
    df_year = df[df['Year'] == year]
    print("Filtered Data:")
    print(df_year.head())
    
    country_dfs = {value: df_year.loc[df_year['Country'] == value] for value in df_year['Country'].unique()}
    country_mean_docs_and_citations = {country: [country_dfs[country]['Documents'].mean(), country_dfs[country]['Citations'].mean()] for country in country_dfs.keys()}
    mean_df = pd.DataFrame.from_dict(country_mean_docs_and_citations, orient='index', columns=['Mean Documents', 'Mean Citations'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Mean Documents', y='Mean Citations', data=mean_df, hue=mean_df.index, ax=ax)

    for country, (mean_docs, mean_citations) in mean_df.iterrows():
        ax.annotate(country, (mean_docs, mean_citations), textcoords="offset points", xytext=(0,10), ha='center')

    ax.set_title(f'Mean Documents vs Mean Citations for Each Country ({year})')
    ax.set_xlabel('Mean Documents')
    ax.set_ylabel('Mean Citations')
    ax.legend([], [], frameon=False)
    ax.grid(True)
    plt.tight_layout()
    return fig


from matplotlib.widgets import Slider

def update_plot(val):
    year = int(year_slider.val)
    fig = make_plot(year)
    plt.draw()

# Create initial plot
initial_year = 2015
fig = make_plot(initial_year)

# Add slider for year selection
slider_ax = plt.axes([0.1, 0.01, 0.8, 0.03])
year_slider = Slider(slider_ax, 'Year', df['Year'].min(), df['Year'].max(), valinit=initial_year, valstep=1)
year_slider.on_changed(update_plot)

plt.show()