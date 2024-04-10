import altair as alt
import matplotlib.pyplot as plt
import streamlit as st


def scatter_plot_for_year(data, year):
    # Filtrer les données pour l'année sélectionnée
    data_year_filtered = data[data['Year'] == year]

    # Calculer les moyennes pour l'année sélectionnée
    grouped_data_for_year = data_year_filtered.groupby('Country').agg(
        avg_documents=('Documents', 'mean'),
        avg_citations=('Citations', 'mean'),
        median_rank=('Rank', 'median'),
        avg_h_index=('H.index', 'mean')
    ).reset_index()

    # Nuage de points pour l'année sélectionnée
    scatter_plot = alt.Chart(grouped_data_for_year).mark_circle().encode(
        x=alt.X('avg_documents', title='Moyenne des documents'),
        y=alt.Y('avg_citations', title='Moyenne des citations'),
        color=alt.Color('median_rank', scale=alt.Scale(scheme='category20b'), title='Rang'),
        size=alt.Size('avg_h_index', title='H-index moyen'),
        tooltip=['Country', 'avg_documents', 'avg_citations', 'median_rank', 'avg_h_index']
    ).properties(
        width=700,
        height=400
    )

    return st.altair_chart(scatter_plot, use_container_width=True)


def plot_top_10_ranks_evolution(top_10_ranks):
    # Filtrer les données pour ne prendre en compte que les 10 premiers rangs
    top_10_ranks = top_10_ranks[top_10_ranks['Rank'] <= 10]

    # Créer un graphique avec Altair
    chart = alt.Chart(top_10_ranks).mark_line().encode(
        x='Year:O',
        y=alt.Y('Rank:O', sort='ascending'),  # Ajouter un pas de 1 à l'axe y
        color='Country:N',
        tooltip=['Country', 'Year', 'Rank']
    ).properties(
        width=600,
        height=400
    ).interactive()

    # Afficher le graphique dans Streamlit
    st.write(chart)


def plot_map(year, h_index, data, world):
    data_filtered = data[(data['Year'] == year) & (data['H.index'] >= h_index)]
    world_filtered = world.merge(data_filtered, how='left', left_on='SOVEREIGNT', right_on='Country')

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.plot(ax=ax, color='black', edgecolor='black')
    world_filtered.plot(ax=ax, column='H.index', cmap='viridis', edgecolor='black', legend=True)

    if not ax.collections:
        vmin = h_index
        vmax = data[
            'H.index'].max()  # Utiliser la valeur maximale de l'ensemble des données comme valeur maximale de l'échelle
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label('Indice H')
    st.pyplot(fig)


def plot_top_countries_map(data, world, top_n, selected_year):
    # Sélectionner uniquement les pays présents dans le top N pour l'année sélectionnée
    top_countries = data[(data['Rank'] <= top_n) & (data['Year'] == selected_year)]

    # Fusionner les données avec les frontières des pays
    merged_data = world.merge(top_countries, how='inner', left_on='ADMIN', right_on='Country')

    # Créer la carte
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plotter les frontières des pays
    world.plot(ax=ax, color='lightgrey')

    # Colorer les pays en fonction du nombre de documents produits
    merged_data.plot(ax=ax, column='Documents', cmap='viridis', legend=True)
    plt.axis('off')

    # Afficher la carte dans Streamlit
    st.pyplot(fig)
