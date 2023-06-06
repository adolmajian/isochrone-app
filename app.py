import folium
import streamlit as st
import osmnx as ox
import networkx as nx
import pydeck as pdk
import geopandas as gpd

from geonetworkx.tools import get_alpha_shape_polygon

from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, MultiPolygon

st.set_page_config(layout="wide")

st.markdown(
    """
<style>

#root > div:nth-child(1) > div.withScreencast > div > div > div > section:nth-child(1) > div:nth-child(1) 
    > div:nth-child(2) > div > div:nth-child(1) > div > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(3) 
    > div > div:nth-child(2) {
        position: absolute;
        margin: 0;
        padding: 0;
        display: flex;
        text-align: center;
        align-content: center;
        justify-content: center;
        align-items: center;
        top: calc(50% - 48px);
        pointer-events: none;
}

#root > div:nth-child(1) > div.withScreencast > div > div > div > section:nth-child(1) > div:nth-child(1) 
    > div:nth-child(2) > div > div:nth-child(1) > div > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(3) 
    > div > div:nth-child(2) > div {
        display: flex;
        align-content: center;
        justify-content: center;
        align-items: center;
}

#root > div:nth-child(1) > div.withScreencast > div > div > div > section:nth-child(1) > div:nth-child(1) 
    > div:nth-child(2) > div > div:nth-child(1) > div > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(3) 
    > div > div:nth-child(2) > div > div > p {
        display: flex;
        text-align: center;
        color: rgb(255, 75, 75);
        font-weight: bold;
        margin: 0;
        font-size: 50px;
        align-content: center;
        justify-content: center;
        align-items: center;
        text-shadow: 5px 5px 5px grey;
}
</style>
""",
    unsafe_allow_html=True
)

# Constants
PALETTE = {
    'red': [255, 75, 75],
    'white': [255, 255, 255],
    'yellow': [255, 249, 127],
    'teal': [0, 108, 103],
    'beige': [255, 235, 198],
    'orange': [255, 177, 0],
    'blue': [0, 56, 68],
    'cyan': [0, 148, 198],
    'green': [46, 204, 113]
}


# Functions
def plug_shape_holes(geom):
    if geom.geom_type.lower() == 'multipolygon':
        return MultiPolygon([Polygon(p.exterior) for p in shape_concave.geoms]).buffer(0)
    if geom.geom_type.lower() == 'polygon':
        return Polygon(geom.exterior)


# Starting variables
center = [45.503032, -73.566424]
zoom = 15
source_ = center
use_time = False

# State variables
if 'center' not in st.session_state:
    st.session_state.center = center

if 'zoom' not in st.session_state:
    st.session_state.zoom = zoom

if "cost_type" not in st.session_state:
    st.session_state.cost_type = 'distance'

# Map
m = folium.Map(location=center, zoom_start=zoom)
fg = folium.FeatureGroup(name="Markers")

# Layout
st.title('Welcome to the Isochrone calculator app!')

tab1, tab2 = st.tabs(["Isochrone Calculator", "About Accessibility"])

# Sidebar Controls
with st.sidebar:
    st.title('Controls')

    with st.form("map_form"):

        # Map Controls
        st.subheader('Choose a location')
        st.caption('The isochrone will be calculated around the point')

        # Create the map
        with st.container():

            # When the user pans the map ...
            map_state_change = st_folium(
                m,
                key="new",
                feature_group_to_add=fg,
                height=300,
                width='100%',
                returned_objects=['center', 'zoom'],
            )

            st.write('⌖')

            if 'center' in map_state_change:
                st.session_state.center = [map_state_change['center']['lat'], map_state_change['center']['lng']]

            if 'zoom' in map_state_change:
                st.session_state.zoom = map_state_change['zoom']

        with st.container():
            col1, col2 = st.columns([2, 1])

            with col1:
                dec = 10
                st.write(round(st.session_state.center[0], dec), ', ', round(st.session_state.center[1], dec))

            with col2:
                # btn_click = st.button('Set location')
                submitted = st.form_submit_button("Set location")
                # if btn_click:
                if submitted:
                    source_ = st.session_state["center"]
                    fg.add_child(folium.Marker(st.session_state["center"]))

    # Distance/Accessibility Controls
    st.subheader('Accessibility Controls')

    # st.radio(
    #     "Distance or travel time threshold 👇",
    #     ["Distance", "Time", ],
    #     key="cost_type",
    #     horizontal=True,
    # )
    distance = st.slider('Select a maximum walking distance (m)', 250, 1000, 500, 50)
    interval = st.slider('Select an interval (m)', 50, 250, 100, 50)
    if interval >= distance:
        st.error('Interval must be smaller than maximum distance', icon="🚨")
    else:
        st.empty()

    st.divider()

    # Shape Controls - Vector
    st.subheader('Shape Approximation Controls')

    st.markdown('#### Concave Shape')
    alpha = st.slider('Select the alpha percentile', 0, 100, 85, 1,
                      help='Applies to the concave hull creation. Will keep "percentile" percent of triangles')
    plug_holes = st.checkbox('Fill in holes', value=False,
                             help='If the resulting shape contains holes, it will plug/remove them')
    force_single = st.checkbox('Force single part shape', value=False,
                               help='If the resulting shape is not a single polygon, it will modify the alpha shape '
                                    'parameter until it becomes a single shape. Will override the alpha percentile '
                                    'parameter.')
    new_alpha = alpha

    if new_alpha != alpha:
        st.write('Overwritten alpha percentile:', new_alpha)

    st.markdown('#### Link Offset Shape')
    offset = st.slider('Select an offset distance (m)', 5, 100, 25, 5,
                       help='Applies to both link offset and node buffers')
    cap_style = st.radio(
        "Buffer cap style",
        ["round", "square", "flat"],
        key="cap_style",
        horizontal=True,
    )
    clip_to_buffer = st.checkbox('Clip shape to buffer', value=False,
                                 help='With the offset method, the resulting shape can overflow the cicular buffer. '
                                      'Checking the box will remove the excess.')

    st.markdown('#### Grid interpolation')
    cell_size = st.slider('Select a resolution for the interpolation (m)', 10, 100, 25, 5)
    algo = st.radio(
        "Interpolation algorithm",
        ["Triangular Irregular Network (TIN)", "Inverse Distance Weighting (IDW)"],
        key="algo",
        horizontal=True,
    )

# Isochrone calculation
with tab1:
    # All map calculations here
    # Build the graph
    graph = ox.graph_from_point(source_, network_type='walk', dist=distance + 250)
    nodes, edges = ox.utils_graph.graph_to_gdfs(graph)

    # Get UTM crs for distance based geoprocessing
    utm_crs = nodes.estimate_utm_crs()

    # Construct GeoDataFrames
    source = gpd.GeoDataFrame(geometry=[Point(source_[::-1])], crs='epsg:4326')
    buffer = source.to_crs(utm_crs).buffer(distance).to_crs('epsg:4326')

    # Merge source to graph
    # gnx.spatial_points_merge(graph, source, inplace=True)

    # Clip
    nodes = nodes.clip(buffer)
    edges = edges.clip(buffer)

    # Viewport
    viewport = pdk.data_utils.compute_view(points=[[p.xy[0][0], p.xy[1][0]] for p in nodes.geometry.to_list()],
                                           view_proportion=0.9)

    # Find the closest node to source_
    start_node = ox.nearest_nodes(graph, source_[1], source_[0])

    # Get start node as gdf for plotting
    start_point = nodes[nodes.index == start_node].copy()

    # if use_time:
    #     # Reproject the graph
    #     graph_utm = ox.project_graph()
    #
    #     # Add travel time as cost
    #     meters_per_minute = travel_speed * 1000 / 60  # km per hour to m per minute
    #     for u, v, k, data in graph.edges(data=True, keys=True):
    #         data['time'] = data['length'] / meters_per_minute

    # Calculate isochrone
    subgraph = nx.ego_graph(graph, start_node, radius=distance, distance='length')
    acc_nodes, acc_edges = ox.utils_graph.graph_to_gdfs(subgraph)

    # Convex shape
    shape_convex = acc_nodes.unary_union.convex_hull
    shape_convex_df = gpd.GeoDataFrame(geometry=[shape_convex], crs='epsg:4326')

    # Concave shape
    pts = list(acc_nodes.to_crs(utm_crs).geometry.apply(lambda p: (p.x, p.y)))
    shape_concave = get_alpha_shape_polygon(pts, alpha)

    # Concave shape - CONDITION - no holes
    if plug_holes:
        shape_concave = plug_shape_holes(shape_concave)

    # Concave shape - CONDITION - single part geometry
    if force_single:
        while shape_concave.geom_type.lower() == 'multipolygon':
            if new_alpha == 100:
                break
            new_alpha += 1
            shape_concave = get_alpha_shape_polygon(pts, new_alpha)
            if plug_holes:
                shape_concave = plug_shape_holes(shape_concave)
            st.write('Overwritten alpha percentile:', new_alpha)

    shape_concave_df = gpd.GeoDataFrame(geometry=[shape_concave], crs=utm_crs).to_crs(
        'epsg:4326')

    # Offset shape
    shape_offset = acc_edges.to_crs(utm_crs).buffer(offset, cap_style='round').unary_union
    shape_offset_df = gpd.GeoDataFrame(geometry=[shape_offset], crs=utm_crs).to_crs(
        'epsg:4326')

    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f'### Walking Network Inside {distance} m Buffer')

            st.pydeck_chart(pdk.Deck(
                initial_view_state=viewport,
                layers=[
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=buffer,
                        get_fill_color=PALETTE['red'] + [25],
                        get_line_color=PALETTE['red'] + [200],
                        line_width_max_pixels=1,
                        stroked=True,
                        filled=True,
                        # pickable=True,
                        auto_highlight=True,
                        # tooltip=True
                    ),
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=source,
                        get_radius=20,
                        get_fill_color=PALETTE['yellow'] + [200],
                        get_line_color=PALETTE['white'] + [100],
                        line_width_max_pixels=3,
                        stroked=True,
                        # filled=True,
                        pickable=True,
                        auto_highlight=True,
                        # tooltip=True
                    ),
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=shape_convex_df,
                        get_fill_color=PALETTE['teal'] + [100],
                        get_line_color=PALETTE['teal'] + [200],
                        line_width_max_pixels=1,
                        stroked=True,
                        filled=True,
                        pickable=True,
                        auto_highlight=True,
                        # extruded=True,
                        # get_elevation=10,
                        # tooltip=True
                    ),
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=shape_concave_df,
                        get_fill_color=PALETTE['cyan'] + [100],
                        get_line_color=PALETTE['beige'] + [200],
                        line_width_max_pixels=1,
                        stroked=True,
                        filled=True,
                        pickable=True,
                        auto_highlight=True,
                        # extruded=True,
                        # get_elevation=30,
                        # tooltip=True
                    ),
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=shape_offset_df,
                        get_fill_color=PALETTE['green'] + [100],
                        get_line_color=PALETTE['blue'] + [200],
                        line_width_max_pixels=1,
                        stroked=True,
                        filled=True,
                        pickable=True,
                        auto_highlight=True,
                        # extruded=True,
                        # get_elevation=50,
                        # tooltip=True
                    ),
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=start_point,
                        get_radius=10,
                        get_fill_color=(255, 137, 93, 255),
                        get_line_color=[255, 255, 255, 100],
                        line_width_max_pixels=3,
                        stroked=True,
                        # filled=True,
                        pickable=True,
                        auto_highlight=True,
                        # tooltip=True
                    ),
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=edges,
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=1,
                        stroked=True,
                        filled=True,
                        pickable=True,
                        auto_highlight=True,
                        # tooltip=True
                    ),
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=acc_edges,
                        get_line_color=[255, 75, 75],
                        line_width_min_pixels=2,
                        stroked=True,
                        filled=True,
                        pickable=True,
                        auto_highlight=True,
                        # tooltip=True
                    ),
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=nodes,
                        get_radius=3,
                        # get_fill_color=[255, 75, 75],
                        get_fill_color=[255, 255, 255],
                        get_line_color=[255, 255, 255, 100],
                        line_width_max_pixels=3,
                        stroked=True,
                        # filled=True,
                        pickable=True,
                        auto_highlight=True,
                        # tooltip=True
                    ),
                    pdk.Layer(
                        type="GeoJsonLayer",
                        data=acc_nodes,
                        get_radius=3,
                        get_fill_color=[255, 75, 75],
                        # get_fill_color=[255, 255, 255],
                        get_line_color=[255, 255, 255, 100],
                        line_width_max_pixels=3,
                        stroked=True,
                        # filled=True,
                        pickable=True,
                        auto_highlight=True,
                        # tooltip=True
                    ),

                ]
            ))
    with col2:
        st.markdown('### Legend')

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
