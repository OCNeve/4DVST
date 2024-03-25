import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, CheckButtons

# Load the CSV data
data = pd.read_csv("dataset.csv", delimiter=";" ,decimal=',')


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


first_plot()