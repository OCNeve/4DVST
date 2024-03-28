import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, CheckButtons, TextBox
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
# Load the CSV data
data = pd.read_csv("dataset.csv", delimiter=";" ,decimal=',')


country_name_mapping = {
        "United States": "United States of America",
        "Russian Federation": "Russia",
        "Hong Kong": "Hong Kong S.A.R.",
        "Czech Republic": "Czechia",
        "South Korea": "Korea, South",
        "Taiwan": "Taiwan*",
        "Egypt": "Egypt, Arab Rep.",
        "Iran": "Iran, Islamic Rep.",
        "Netherlands Antilles": "Netherlands",
        "Palestine ": "West Bank and Gaza",
        "Syrian Arab Republic": "Syria",
        "Côte d'Ivoire": "Cote d'Ivoire",
        "Congo": "Congo, Dem. Rep.",
        "Democratic Republic Congo": "Congo, Rep.",
        "Viet Nam": "Vietnam",
        "Moldova": "Moldova, Rep.",
        "North Korea": "Korea, North",
        "Falkland Islands (Malvinas)": "Falkland Islands",
        "French Southern Territories": "French Southern and Antarctic Lands",
        "Saint Helena": "Saint Helena, Ascension and Tristan da Cunha"
    }



def first_plot():
    # Calculate mean documents and citations per country per year
    mean_data = data.groupby(["Year", "Country"]).mean().reset_index()

    # Initialize the seaborn style
    sns.set(style="whitegrid")
    # Create a color map for countries
    unique_countries = mean_data['Country'].unique()
    country_colors = sns.color_palette("Set2", len(unique_countries))
    country_color_map = {country: color for country, color in zip(unique_countries, country_colors)}

    # Create a function to update the plot based on the selected year
    def update(val):
        year = int(slider.val)
        selected_data = mean_data[mean_data["Year"] == year]
        ax.clear()
        scatter = sns.scatterplot(x="Documents", y="Citations", data=selected_data, hue="Country", ax=ax, palette=country_color_map, legend=False)
        ax.set_xlabel("Mean Documents")
        ax.set_ylabel("Mean Citations")
        ax.set_title(f"Mean Documents vs Mean Citations per Country in {year}")

        # Annotate each point with the country name if the checkbox is checked
        if show_labels.get_status()[0]:
            for _, row in selected_data.iterrows():
                ax.annotate(row["Country"], (row["Documents"], row["Citations"]), textcoords="offset points", xytext=(0,10), ha='center')
        else:
            scatter.legend([], [], frameon=False)
        fig.canvas.draw_idle()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    checkbox_ax = plt.axes([0.1, 0.03, 0.15, 0.03])
    show_labels = CheckButtons(checkbox_ax, ['Show Labels'], [False])
    show_labels.on_clicked(update)
    # Create a slider for selecting the year
    slider_ax = plt.axes([0.1, 0.0, 0.8, 0.03])
    slider = Slider(slider_ax, 'Year', data['Year'].min(), data['Year'].max(), valinit=data['Year'].min(), valstep=1)
    slider.on_changed(update)

  
    update(None)

    plt.show()




def second_plot():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))  # Provide the path to the downloaded shapefile

    # Load the dataset containing country ranks over years
    rank_data = pd.read_csv("dataset.csv", delimiter=";", decimal=',')
    
    rank_data['Country'] = rank_data['Country'].map(country_name_mapping).fillna(rank_data['Country'])

    # Function to update the plot based on the selected year
    def update(val):
        year = int(slider.val)
        year_data = rank_data[rank_data['Year'] == year]
        top10_countries = year_data.nsmallest(10, 'Rank')['Country']

        world_merged = world.merge(year_data, left_on='name', right_on='Country')
        world_merged['color'] = np.where(world_merged['Country'].isin(top10_countries), world_merged['Rank'], 12)
        ax.clear()
        world_merged.plot(column='color', cmap='coolwarm', ax=ax,
                          edgecolor='k', linewidth=0.3, legend=False)
        ax.set_title(f"Country Ranks in {year}")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        # Update country labels visibility based on the checkbox state
        if show_labels.get_status()[0]:
            for idx, row in world_merged.iterrows():
                ax.text(row.geometry.centroid.x, row.geometry.centroid.y, row['Country'], fontsize=8)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a slider for selecting the year
    slider_ax = plt.axes([0.1, 0.02, 0.8, 0.03])
    slider = Slider(slider_ax, 'Year', rank_data['Year'].min(), rank_data['Year'].max(),
                    valinit=rank_data['Year'].min(), valstep=1)
    slider.on_changed(update)

    # Create a checkbox for toggling country name labels
    checkbox_ax = plt.axes([0.1, 0.95, 0.15, 0.03])
    show_labels = CheckButtons(checkbox_ax, ['Show Labels'], [False])
    show_labels.on_clicked(update)

    # Initial plot
    update(None)

    # Create legend
    top10_patch = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=1, vmax=11))
    top10_patch.set_array([])
    top10_legend = plt.colorbar(top10_patch, ax=ax, orientation='horizontal', label='Country Rank')
    top10_legend.set_ticks(range(1, 12))
    top10_legend.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Rank > 10'])

    plt.show()



place_legend = True
def third_plot():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    # Load the dataset containing average H-index per country
    h_index_data = pd.read_csv("dataset.csv", delimiter=";", decimal=',')

    h_index_data['Country'] = h_index_data['Country'].map(country_name_mapping).fillna(h_index_data['Country'])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sliders for selecting the threshold value and the year
    threshold_slider_ax = plt.axes([0.1, 0.02, 0.8, 0.03])
    threshold_slider = Slider(threshold_slider_ax, 'Threshold', h_index_data['H.index'].min(), h_index_data['H.index'].max()-1, valinit=h_index_data['H.index'].min() , valstep=1)

    

    # Create a checkbox for toggling country name labels
    checkbox_ax = plt.axes([0.1, 0.95, 0.15, 0.03])
    show_labels = CheckButtons(checkbox_ax, ['Show Labels'], [False])

    def update(val):
        global place_legend
        threshold = threshold_slider.val
        filtered_data = h_index_data[(h_index_data['H.index'] > threshold)  &
                                     (h_index_data['Country'])]

        world_merged = world.merge(filtered_data, left_on='name', right_on='Country')
        ax.clear()
        world.plot(ax=ax, color='lightgrey', edgecolor='k')  # Plot all countries in light grey
        if place_legend:
            world_merged.plot(ax=ax, column='H.index', cmap='viridis', legend=True,
                              legend_kwds={'label': "Average H-index"}, edgecolor='k', linewidth=0.3)
            place_legend = False
        else:
            world_merged.plot(ax=ax, column='H.index', cmap='viridis', edgecolor='k', linewidth=0.3)

        ax.set_title(f"Countries with Average H-index > {threshold}")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        # Update country labels visibility based on the checkbox state
        if show_labels.get_status()[0]:
            for idx, row in world_merged.iterrows():
                ax.text(row.geometry.centroid.x, row.geometry.centroid.y, row['Country'], fontsize=8)

        # Remove existing legend if any
        legend = ax.get_legend()
        if legend:
            legend.remove()  # Remove existing legend if any
            plt.colorbar(ax.collections[0], ax=ax, orientation='horizontal', label='Average H-index')

    threshold_slider.on_changed(update)
    show_labels.on_clicked(update)
    update(None)
    plt.show()



def fourth_plot():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Function to get the top countries for a given year
    def get_top_countries(year):
        mean_data = data.groupby(["Year", "Country"]).mean().reset_index()
        selected_data = mean_data[mean_data["Year"] == year]
        top_countries = selected_data.nlargest(5, 'Documents')['Country'].tolist()  # Adjust the number of top countries as needed
        return top_countries

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sliders for selecting the year
    year_slider_ax = plt.axes([0.1, 0.02, 0.8, 0.03])
    year_slider = Slider(year_slider_ax, 'Year', data['Year'].min(), data['Year'].max(), valinit=data['Year'].min(), valstep=1)

    # Create a checkbox for toggling country name labels
    checkbox_ax = plt.axes([0.1, 0.95, 0.15, 0.03])
    show_labels = CheckButtons(checkbox_ax, ['Show Labels'], [False])

    place_legend = True  # Variable to track if the legend has been placed

    def update(val):
        nonlocal place_legend
        year = int(year_slider.val)
        top_countries = get_top_countries(year)
        
        # Filter the data for the selected year and top countries
        filtered_data = data[(data['Year'] == year) & (data['Country'].isin(top_countries))]

        world_merged = world.merge(filtered_data, left_on='name', right_on='Country')
        ax.clear()
        world.plot(ax=ax, color='lightgrey', edgecolor='k')  # Plot all countries in light grey
        world_merged.plot(ax=ax, column='Documents', cmap='plasma', edgecolor='k', linewidth=0.3)
        
        ax.set_title(f"Top Countries' Documents in {year}")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        if show_labels.get_status()[0]:
            for idx, row in world_merged.iterrows():
                ax.text(row.geometry.centroid.x, row.geometry.centroid.y, str(row['Documents']), fontsize=8)

        # Add legend only once
        if place_legend:
            # Use the same colormap and range for the color bar
            plt.colorbar(ax.collections[0], ax=ax, orientation='horizontal', label='Number of Documents', cmap='plasma')
            place_legend = False

    year_slider.on_changed(update)
    show_labels.on_clicked(update)
    update(None)

    plt.show()



first_plot()
second_plot()
third_plot()
fourth_plot()




