import streamlit as st
from utils import scatter_plot_for_year, plot_top_10_ranks_evolution, plot_map, plot_top_countries_map
import pandas as pd
import geopandas as gpd

data = pd.read_csv('data/publications_scientifiques_par_pays.csv')
data.loc[data['Country'] == 'United States', 'Country'] = 'United States of America'
data.loc[data['Country'] == 'Russian Federation', 'Country'] = 'Russia'
world = gpd.read_file('data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')

st.set_page_config(layout="wide")

st.sidebar.title('Filtres')
selected_year_slider_key = "selected_year_slider"
selected_year = st.sidebar.slider('Sélectionner une année', min_value=data['Year'].min(), max_value=data['Year'].max(),
                                  value=data['Year'].min(), key=selected_year_slider_key)

tabs = st.tabs(["Nuage de Points", "Évolution des Rangs", "Carte Top 10", "Carte H-index"])

with tabs[0]:
    st.markdown(
        "<h2 style='font-weight:bold;margin-bottom:20px;font-size:35px'>Nuage de Points: Documents vs Citations pour " + str(
            selected_year) + "</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        scatter_plot_for_year(data, selected_year)
    with col2:
        st.subheader('Le leader :')
        st.write(
            "On peut constater à travers ce graphique que les États-Unis sont, de 1996 à 2014, le pays qui arrive à produire le plus de documents qui sont "
            "par la suite utilisés pour la recherche.")
        st.write("")
        st.subheader('L\'ascension de la Chine :')
        st.write(
            "On remarque également que la Chine, à partir de 2005, augmente considérablement sa production de documents "
            "et cela jusqu'en 2014. Ces documents produits ne sont pas autant cités que ceux des États-Unis mais dépassent les citations de documents des pays européens en 2011.")

with tabs[1]:
    st.markdown("<h2 style='font-weight:bold;margin-bottom:20px;font-size:35px'>Évolution des Rangs des Pays</h2>",
                unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        top_10_ranks = data[data['Rank'] <= 10]
        plot_top_10_ranks_evolution(top_10_ranks)
    with col2:
        st.write(
            "Comme nous pouvons le constarter, la Chine entre 1996 et 2014 à fait une augmentation remarquable en ce qui s'agit du rank de production scientifique.")

with tabs[2]:
    st.markdown(
        "<h2 style='font-weight:bold;margin-bottom:20px;font-size:35px'>Carte des pays du top 10 pour l\'année " + str(
            selected_year) + " selon le nombre de documents produits</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        plot_top_countries_map(data, world, 10, selected_year)
    with col2:
        st.write(
            "On déduit à partir de ce graphique que le classement du top 10 des pays reste le même de 1996 à 2014. "
            "Sauf à un moment où l'Australie intègre ce classement à la 10ème position entre 2004 et 2007 à la place de la Russie. "
            "Puis la 10ème position est prise en 2008 par l'Inde et cela jusqu'en 2014.")

with tabs[3]:
    h_index_slider_key = "h_index_threshold_slider"
    h_index_threshold = st.sidebar.slider('Sélectionner un seuil pour l\'indice H', min_value=0,
                                          max_value=data['H.index'].max(), value=0, key=h_index_slider_key)
    st.markdown(
        "<h2 style='font-weight:bold;margin-bottom:20px;font-size:35px'>Carte des pays avec un H-index supérieur à " + str(
            h_index_threshold) + " pour l'année " + str(selected_year) + "</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1], gap="large")
    with col1:
        plot_map(selected_year, h_index_threshold, data, world)
    with col2:
        st.write(
            "On se rend compte qu'à partir de ce graphique que la plupart des pays occidentaux sont performants en termes de production scientifique et d'impact."
            "En Asie, on remarque que les pays les plus productifs sont la Chine, l'Inde, le Japon et la Russie."
            "En Amérique, les États-Unis sont leaders suivis par le Canada, le Groenland, le Brésil, le Mexique et l'Argentine. Le reste des pays est moins productif, notamment en Amérique latine."
            "Concernant l'Afrique, les pays sont peu performants en production de documents et ces derniers ne sont pas énormément cités, sauf pour l'Afrique du Sud.")