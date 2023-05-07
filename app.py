import folium
import streamlit as st

from operator import itemgetter
from folium.plugins import Draw
from streamlit_folium import st_folium


def dict_to_latlng(d):
    return [d['lat'], d['lng']]


# State variables
if 'center' not in st.session_state:
    st.session_state.center = [45.503032, -73.566424]

if 'zoom' not in st.session_state:
    st.session_state.zoom = 15

# if 'location' not in st.session_state:
#     st.session_state.location = folium.Marker(st.session_state.center)

# if 'fg' not in st.session_state:
#     st.session_state.fg = folium.FeatureGroup(name="Markers")

m = folium.Map(location=st.session_state.center, zoom_start=st.session_state.zoom)
fg = folium.FeatureGroup(name="Markers")
fg.add_child(folium.Marker(st.session_state.center))

# Layout
st.set_page_config(layout="wide")

st.title('Welcome to the Isochrone calculator app!')

tab1, tab2 = st.tabs(["Isochrone Calculator", "About Accessibility"])

# Sidebar Controls
with st.sidebar:
    st.header('Controls')

    # Map Controls
    st.subheader('Choose a location')
    st.caption('The isochrone will be calculated around the point')

    # Create the map

    # fg = folium.FeatureGroup(name="Markers")
    # fg.add_child(st.session_state.location)
    # st.session_state.fg.add_child(st.session_state.location)
    # st.session_state.fg.add_child(folium.Marker(st.session_state.center))

    # When the user clicks
    map_state_change = st_folium(
        m,
        feature_group_to_add=fg,
        height=400,
        width='100%',
        returned_objects=['last_clicked', 'zoom', 'bounds', 'center'],
    )
    print(map_state_change)

    if map_state_change['last_clicked']:
        # try:
        center = dict_to_latlng(map_state_change['last_clicked'])
        # st.session_state.center = [loc['lat'], loc['lng']]
        # st.session_state.center = [loc['lat'], loc['lng']]
        # st.session_state.location = folium.Marker(st.session_state.center)
        # s.add_child(st.session_state.location)
        fg.add_child(folium.Marker(center))
    # except Exception as e:
    #     pass

    # st.write(st.session_state.center)

    # Distance Controls
    st.subheader('Accessibility Controls')
    distance = st.slider('Select a maximum walking distance (m)', 250, 1000, 500, 50)
    st.write("I'm ", distance, 'years old')

    st.subheader('Choose a location')

# Isochrone calculation
with tab1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

# Explanations
with tab2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    st.markdown("""
      A common concept in Transportation and Urban Planning is the concept of accessibility; how accessible are amenities or services to a reference(s) point(s). For example, 
      - What percentage of households are within a 5 minute drive to a pharmacy? or 
      - A Transportation Master Plan goal that aims to have a 100% of households within 500 m distance walk from a bus stop by redesigning their bus network by the next 2 years.
      - How many people live 20 minutes, 30 minutes, 45 minutes, 1 hour, and more than an hour away from the university by cycling or public transportation.

      These questions often translate to accessibility or reach map analyses that aim to create a series of `isochrones` (lines/areas or equal travel time) or `isodistances` (lines/areas of equal distance); similar to topo maps with contour lines (lines of equal elevation). The age old question remains however, what is correct way to represent those lines or areas?

      In the case of elevation and topo maps, the elevation model (dgitial evelation model (DEM) or derived topo map) represents an actual continuous surface in the real world. In the case of accessibility, the data points are limited to nodes and links that represent the driving/cycling/walking network and not a surface; i.e. if you want to increase your precision in topo maps, you can increase the sample of points but in the case of network analysis, increasing the number of points will not change the results since the points are still bound to the links that represent the real world. Thus, there is a choice to be made on how to transform points into a line/polygon and that choice will depend on the use case and also on the general state (connectivity, centrality of the network). One choice of algorithm for a chunk of network in an urban area might not be suitable for a chunk of network in a rural more sparse area.

      This app attempts to illustrate the advantages and pitfalls of different algorithms to create lines/polygons of equal time or distance. There are two groups of approaches:
      1. Computational Geometry (working with vector data)
      2. Spatial Interpolation (working in the realm of raster data and then transforming to vector if necessary)

      The important thing to understand is that 
      """)