import warnings

import geopandas as gpd

import folium
import streamlit as st
import osmnx as ox
import networkx as nx
import pydeck as pdk

import matplotlib.pyplot as plt

from geonetworkx.tools import get_alpha_shape_polygon
import geonetworkx as gnx

from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.errors import ShapelyDeprecationWarning

from utils import plug_shape_holes, grid_interpolate, colorize_image, convert_image_to_bytes_url, get_gdf_corners, \
    prepare_interpolation_points, extract_contours_from_singleband_raster
from utils import prepare_buffer_layer, prepare_source_layer, prepare_start_point_layer, prepare_edges_layer, \
    prepare_acc_edges_layer, prepare_nodes_layer, prepare_acc_nodes_layer
from utils import PALETTE

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

st.set_page_config(layout="wide", page_title='Interactive Isochrone Calculator',
                   # description='A web app to calculate isochrones for anywhere in the world using a variety of methods. ',
                   page_icon=":world_map:")

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

# Starting variables
center = [45.503032, -73.566424]
zoom = 15
use_time = False
colormap = plt.cm.YlOrRd
padding_geo = 1 / 111111 * 10
padding = 10
interpolation_algo_dict = {
    'idw': 'Inverse Distance Weighting (IDW)',
    'tin': 'Triangular Irregular Network (TIN)'
}
is_extended_graph = True

# State variables
if 'center' not in st.session_state:
    st.session_state.center = center

if 'source' not in st.session_state:
    st.session_state.source = center

if 'zoom' not in st.session_state:
    st.session_state.zoom = zoom

if "cost_type" not in st.session_state:
    st.session_state.cost_type = 'distance'

# Map
m = folium.Map(location=center, zoom_start=zoom)

# Layout
st.title('Welcome to the Isochrone calculator app!')

tab1, tab2 = st.tabs(["Isochrone Calculator", "How to Use the App"])

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
                submitted = st.form_submit_button("Set location")
                if submitted:
                    st.session_state.source = st.session_state.center

    # Distance/Accessibility Controls
    st.markdown('## Accessibility Controls')

    distance = st.slider('Select a maximum walking distance (m)', 250, 1000, 500, 50)
    interval = st.slider('Select a walking distance interval (m)', min_value=50, max_value=250, value=100, step=50,
                         help='Applies to grid interpolation to create contour lines.')
    if interval >= distance:
        st.error('Interval must be smaller than maximum distance', icon="🚨")
    else:
        st.empty()

    st.divider()

    # Shape Controls - Vector
    st.markdown('## Shape Approximation Controls')

    st.markdown('#### Concave Shape')
    alpha = st.slider('Select the alpha percentile', 0, 100, 85, 1,
                      help='Applies to the concave hull creation. Will keep "percentile" percent of triangles')
    plug_holes_concave = st.checkbox('Fill in holes', value=False,
                                     help='If the resulting shape contains holes, it will plug/remove them. Applies to '
                                          'concave shape.')
    force_single = st.checkbox('Force single part shape', value=False,
                               help='If the resulting shape is not a single polygon, it will modify the alpha shape '
                                    'parameter until it becomes a single shape. Will override the alpha percentile '
                                    'parameter.')
    force_links = st.checkbox('Must contain all edges', value=False,
                              help='Will ensure that all accessible (red) links are within the resulting shape. '
                                   'Will override the alpha percentile '
                                   'parameter.')
    new_alpha = None

    st.markdown('#### Link Offset Shape')
    offset = st.slider('Select an offset distance (m)', 5, 100, 25, 5,
                       help='Applies to link offset')
    cap_style = st.radio(
        "Buffer endcap style",
        ["round", "square", "flat"],
        key="cap_style",
        horizontal=True,
    )
    clip_to_buffer = st.checkbox('Clip shape to theoretical accessibility area', value=False,
                                 help='With the offset method, the resulting shape can overflow the circular buffer. '
                                      'Checking the box will remove the excess.')
    plug_holes_offset = st.checkbox('Fill in holes', value=False,
                                    help='If the resulting shape contains holes, it will plug/remove them. '
                                         'Applies to link offset.')

    st.markdown('#### Grid interpolation')
    cell_size = st.slider('Select a resolution (cell size) for the interpolation (m)', 5, 20, 10, 5)
    algo = st.radio(
        "Interpolation algorithm",
        options=list(interpolation_algo_dict.keys()),
        format_func=lambda x: interpolation_algo_dict[x],
        key="algo",
        horizontal=True,
    )

# Isochrone calculation
with tab1:
    # All map calculations here
    # Build the graph
    try:
        graph = ox.graph_from_point(st.session_state.source, network_type='walk', dist=distance + 250)
        nodes, edges = ox.utils_graph.graph_to_gdfs(graph)
    except Exception as e:
        print(e)
        with st.empty():
            st.subheader(
                ':warning: Was unable to either download OSM data for the chosen location or to construct a suitable '
                'network. Make sure to choose an area that has a street network.')

    # Get UTM crs for distance based geoprocessing
    utm_crs = nodes.estimate_utm_crs()

    # Construct GeoDataFrames
    source = gpd.GeoDataFrame(geometry=[Point(st.session_state.source[::-1])], crs='epsg:4326')
    buffer = source.to_crs(utm_crs).buffer(distance).to_crs('epsg:4326')
    buffer_area = round(source.to_crs(utm_crs).buffer(distance).area.iloc[0] / 1000000, 2)

    # Merge source to graph
    # gnx.spatial_points_merge(graph, source, inplace=True)

    # Clip
    nodes = nodes.clip(buffer)
    edges = edges.clip(buffer)

    # Viewport
    viewport = pdk.data_utils.compute_view(points=[[p.xy[0][0], p.xy[1][0]] for p in nodes.geometry.to_list()],
                                           view_proportion=0.9)

    # Find the closest node to source
    start_node = ox.nearest_nodes(graph, st.session_state.source[1], st.session_state.source[0])

    # Get start node as gdf for plotting
    start_point = nodes[nodes.index == start_node].copy()

    # Calculate isochrone
    try:
        # Extended ego graph
        geo_graph = gnx.GeoGraph(crs=4326)
        geo_graph.add_edges_from_gdf(edges.reset_index(), edge_first_node_attr='u', edge_second_node_attr='v')
        geo_graph.add_nodes_from_gdf(nodes.reset_index(), node_index_attr='osmid')

        subgraph = gnx.extended_ego_graph(geo_graph, start_node, distance, distance='length')
        acc_nodes = subgraph.nodes_to_gdf()
        acc_edges = subgraph.edges_to_gdf()
    except Exception as e:
        print(e)
        # Simple ego graph
        is_extended_graph = False
        subgraph = nx.ego_graph(graph, start_node, radius=distance, distance='length')
        acc_nodes, acc_edges = ox.utils_graph.graph_to_gdfs(subgraph)

    # Convex shape
    shape_convex = acc_nodes.unary_union.convex_hull.buffer(padding_geo)
    shape_convex_df = gpd.GeoDataFrame(geometry=[shape_convex], crs='epsg:4326')
    shape_convex_area = round(shape_convex_df.to_crs(utm_crs).iloc[0].geometry.area / 1000000, 2)

    # Concave shape
    pts = list(acc_nodes.to_crs(utm_crs).geometry.apply(lambda p: (p.x, p.y)))
    shape_concave = get_alpha_shape_polygon(pts, alpha)

    # Concave shape - CONDITION - no holes
    if plug_holes_concave:
        shape_concave = plug_shape_holes(shape_concave).buffer(padding)

    # Concave shape - CONDITION - single part geometry
    if force_single or force_links:
        # Check multi
        is_multi = shape_concave.geom_type.lower() == 'multipolygon'

        # Check containment
        is_contained = True  # This is to save time but makes for complicated code. hmm...
        if force_links:
            links_dissolved = acc_edges.to_crs(utm_crs).dissolve().geometry.iloc[0]
            is_contained = shape_concave.contains(links_dissolved)

        if is_multi or not is_contained:
            new_alpha = alpha

        while is_multi or not is_contained:
            if new_alpha == 100:
                break
            new_alpha += 1

            shape_concave = get_alpha_shape_polygon(pts, new_alpha).buffer(padding)
            is_multi = shape_concave.geom_type.lower() == 'multipolygon'

            if force_links:
                is_contained = shape_concave.contains(links_dissolved)

            if plug_holes_concave:
                shape_concave = plug_shape_holes(shape_concave)

    # shape_concave = shape_concave.buffer(padding)
    shape_concave_area = round(shape_concave.area / 1000000, 2)
    shape_concave_df = gpd.GeoDataFrame(geometry=[shape_concave], crs=utm_crs).to_crs(
        'epsg:4326')

    # Offset shape
    shape_offset = acc_edges.to_crs(utm_crs).buffer(offset, join_style='round', cap_style=cap_style).unary_union

    # Offset shape - CONDITION - no holes
    if plug_holes_offset:
        shape_offset = plug_shape_holes(shape_offset)

    # Offset shape - CONDITION - clip
    if clip_to_buffer:
        shape_offset = shape_offset.intersection(buffer.to_crs(utm_crs).geometry.iloc[0])

    shape_offset_area = round(shape_offset.area / 1000000, 2)
    shape_offset_df = gpd.GeoDataFrame(geometry=[shape_offset], crs=utm_crs).to_crs(
        'epsg:4326')

    # Prepare points layer with column with values to interpolate
    node_dists = prepare_interpolation_points(subgraph, start_node, acc_nodes, colormap)

    # Perform the interpolation
    if algo == 'idw':
        img_singleband, ds, transform = grid_interpolate(node_dists, buffer.to_crs(utm_crs), cell_size, algo='idw')
    if algo == 'tin':
        img_singleband, ds, transform = grid_interpolate(node_dists, buffer.to_crs(utm_crs), cell_size, algo='tin')

    # Colorize and get RGBA uint8 image
    img_rgba = colorize_image(img_singleband, colormap=colormap)

    # Convert image to encoded bytes
    img_rgba_encoded_url = convert_image_to_bytes_url(img_rgba)
    img_bounds = get_gdf_corners(buffer)

    # Extact contours
    contours, _ = extract_contours_from_singleband_raster(img_singleband, ds, distance, interval, colormap)
    # contours = contours.clip(buffer.to_crs(utm_crs).buffer(-10).to_crs('epsg:4326'))

    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            inner_col1, inner_col2 = st.columns(2)

            buffer_layer = prepare_buffer_layer(buffer)
            source_layer = prepare_source_layer(source)
            start_point_layer = prepare_start_point_layer(start_point)
            edges_layer = prepare_edges_layer(edges)
            acc_edges_layer = prepare_acc_edges_layer(acc_edges)
            nodes_layer = prepare_nodes_layer(nodes)
            acc_nodes_layer = prepare_acc_nodes_layer(acc_nodes)

            with inner_col1:
                # Buffer + Convex
                st.markdown(f'### Buffer vs Convex Shape')
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=viewport,
                    layers=[
                        buffer_layer,
                        pdk.Layer(
                            type="GeoJsonLayer",
                            data=shape_convex_df,
                            # get_fill_color=PALETTE['teal'] + [100],
                            # get_line_color=PALETTE['teal'] + [200],
                            get_fill_color=PALETTE['aquamarine'] + [100],
                            get_line_color=PALETTE['aquamarine'] + [200],
                            line_width_max_pixels=1,
                            stroked=True,
                            filled=True,
                            pickable=True,
                            auto_highlight=True,
                        ),
                        source_layer,
                        start_point_layer,
                        edges_layer,
                        acc_edges_layer,
                        nodes_layer,
                        acc_nodes_layer
                    ]
                ))

                # Buffer + Offset
                st.markdown(f'### Buffer vs Offset Shape')
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=viewport,
                    layers=[
                        buffer_layer,
                        pdk.Layer(
                            type="GeoJsonLayer",
                            data=shape_offset_df,
                            get_fill_color=PALETTE['purple'] + [100],
                            get_line_color=PALETTE['beige'] + [200],
                            line_width_max_pixels=1,
                            stroked=True,
                            filled=True,
                            pickable=True,
                            auto_highlight=True,
                        ),
                        source_layer,
                        start_point_layer,
                        edges_layer,
                        acc_edges_layer,
                        nodes_layer,
                        acc_nodes_layer
                    ]
                ))

            with inner_col2:
                # Buffer + Concave
                st.markdown(f'### Buffer vs Concave Shape')
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=viewport,
                    layers=[
                        buffer_layer,
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
                        source_layer,
                        start_point_layer,
                        edges_layer,
                        acc_edges_layer,
                        nodes_layer,
                        acc_nodes_layer
                    ]
                ))

                # Buffer + Grid Interpolation
                st.markdown(f'### {algo.upper()} Grid Interpolation + Contours')
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=viewport,
                    tooltip={
                        "html": "<b>Contour value:</b> {contour}",
                        "style": {
                            "backgroundColor": "steelblue",
                            "color": "white"
                        }
                    },
                    layers=[
                        pdk.Layer(
                            type="BitmapLayer",
                            data=None,
                            image=img_rgba_encoded_url,
                            bounds=img_bounds,
                            opacity=0.5
                        ),
                        buffer_layer,
                        source_layer,
                        start_point_layer,
                        edges_layer,
                        acc_edges_layer,
                        nodes_layer,
                        pdk.Layer(
                            type="GeoJsonLayer",
                            data=node_dists,
                            get_radius=8,
                            get_fill_color='color_rgb',
                            get_line_color=[255, 255, 255, 175],
                            # line_width_min_pixels=1,
                            line_width_max_pixels=4,
                            stroked=True,
                            # filled=True,
                            pickable=True,
                            auto_highlight=True,
                            tooltip=False
                        ),
                        pdk.Layer(
                            type="GeoJsonLayer",
                            data=contours,
                            # get_line_color='color_rgb',
                            get_line_color=[255, 255, 255, 175],
                            pickable=True,
                            filled=True,
                            stroked=True,
                            # auto_highlight=True,
                            line_width_min_pixels=6,
                            line_width_max_pixels=9,
                            tooltip=True,

                        ),
                        pdk.Layer(
                            type="GeoJsonLayer",
                            data=contours,
                            get_line_color='color_rgb',
                            # get_line_color=[255, 255, 255, 175],
                            pickable=True,
                            filled=True,
                            stroked=True,
                            auto_highlight=True,
                            line_width_min_pixels=3,
                            line_width_max_pixels=6,
                        ),
                    ]
                ))
    with col2:
        st.markdown('### Legend')

        #  Shapes
        st.markdown('#### Shapes')
        st.markdown(f"""
            <span style="display: inline-flex; align-items: center;">
                <span style="background-color: rgba(255, 75, 75, 0.1); border: 1px solid rgba(255, 75, 75, 0.78); display: inline-block; width: 20px; height: 20px;"></span>
                <span style="padding-left: 5px;">Buffer ({buffer_area} km<sup>2</sup>)</span>
            </span>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <span style="display: inline-flex; align-items: center;">
                <span style="background-color: rgba(165, 255, 214, 0.4); border: 1px solid rgba(165, 255, 214, 0.78); display: inline-block; width: 20px; height: 20px;"></span>
                <span style="padding-left: 5px;">Convex Shape ({shape_convex_area} km<sup>2</sup>)</span>
            </span>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <span style="display: inline-flex; align-items: center;">
                <span style="background-color: rgba(0, 148, 198, 0.4); border: 1px solid rgba(255, 235, 198, 0.78); display: inline-block; width: 20px; height: 20px;"></span>
                <span style="padding-left: 5px;">Concave Shape ({shape_concave_area} km<sup>2</sup>)</span>
            </span>
            """, unsafe_allow_html=True)
        st.markdown(f"""
                    <span style="display: inline-flex; align-items: center;">
                        <span style="background-color: rgba(136, 67, 204, 0.4); border: 1px solid rgba(255, 235, 198, 0.78); display: inline-block; width: 20px; height: 20px;"></span>
                        <span style="padding-left: 5px;">Offset Shape ({shape_offset_area} km<sup>2</sup>)</span>
                    </span>
                    """, unsafe_allow_html=True)

        # User input
        st.markdown('#### User input')
        st.markdown(f"""
                    <span style="display: inline-flex; align-items: center;">
                        <span style="background-color: rgba(255, 249, 127, 0.78); border: 3px solid rgba(255, 255, 255, 0.4); border-radius: 50%; display: inline-block; width: 14px; height: 14px; margin: 0px 3px 0px 3px;"></span>
                        <span style="padding-left: 5px;">User selected center</span>
                    </span>
                    """, unsafe_allow_html=True)
        st.write('Overwritten alpha percentile:', new_alpha)

        # Network
        st.markdown('#### Network')
        st.markdown(f"""
                    <span style="display: inline-flex; align-items: center;">
                        <span style="background-color: rgba(255, 137, 93, 0.78); border: 3px solid rgba(255, 255, 255, 0.4); border-radius: 50%; display: inline-block; width: 14px; height: 14px; margin: 0px 3px 0px 3px;"></span>
                        <span style="padding-left: 5px;">Closest node to center</span>
                    </span>
                    """, unsafe_allow_html=True)
        st.markdown(f"""
                    <span style="display: inline-flex; align-items: center;">
                        <span style="background-color: rgba(255, 255, 255, 1.0); border: 2px solid rgba(255, 255, 255, 0.4); border-radius: 50%; display: inline-block; width: 10px; height: 10px; margin: 0px 5px 0px 5px;"></span>
                        <span style="padding-left: 5px;">Inaccessible nodes</span>
                    </span>
                    """, unsafe_allow_html=True)
        st.markdown(f"""
                    <span style="display: inline-flex; align-items: center;">
                        <span style="background-color: rgba(255, 255, 255, 1.0); border-radius: 75%; display: inline-block; width: 20px; height: 1px; margin-top: 1px;"></span>
                        <span style="padding-left: 5px;">Inaccessible links</span>
                    </span>
                    """, unsafe_allow_html=True)
        st.markdown(f"""
                    <span style="display: inline-flex; align-items: center;">
                        <span style="background-color: rgba(255, 75, 75, 1.0); border: 2px solid rgba(255, 255, 255, 0.9); border-radius: 50%; display: inline-block; width: 10px; height: 10px; margin: 0px 5px 0px 5px;"></span>
                        <span style="padding-left: 5px;">Accessible nodes</span>
                    </span>
                    """, unsafe_allow_html=True)
        st.markdown(f"""
                    <span style="display: inline-flex; align-items: center;">
                        <span style="background-color: rgba(255, 75, 75, 1.0); border-radius: 75%; display: inline-block; width: 20px; height: 1px; margin-top: 1px;"></span>
                        <span style="padding-left: 5px;">Accessible links</span>
                    </span>
                    """, unsafe_allow_html=True)
        st.write('Is extended ego graph:', is_extended_graph)

        st.markdown('#### Contours')
        st.markdown(f"""
                    <span style="display: flex; align-items: center;">
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); grid-gap: 2px;">
                      <div style="background-color: rgba(219, 219, 170, 1.0); border: 0.5px solid rgba(255, 255, 255, 0.9); border-radius: 50%; display: inline-block; width: 9px; height: 9px;"></div>
                      <div style="background-color: rgba(218, 174, 78, 1.0); border: 0.5px solid rgba(255, 255, 255, 0.9); border-radius: 50%; display: inline-block; width: 9px; height: 9px;"></div>
                      <div style="background-color: rgba(207, 29, 9, 1.0); border: 0.5px solid rgba(255, 255, 255, 0.9); border-radius: 50%; display: inline-block; width: 9px; height: 9px;"></div>
                      <div style="background-color: rgba(104, 0, 11, 1.0); border: 0.5px solid rgba(255, 255, 255, 0.9); border-radius: 50%; display: inline-block; width: 9px; height: 9px;"></div>
                    </div>
                      <div style="margin-left: 10px;">Nodes colored by access distance</div>
                    </span>
                    """, unsafe_allow_html=True)
        st.markdown(f"""
                    <span style="display: inline-flex; align-items: center;">
                        <img style="display: inline-block; width=20px; height=20px; margin=0; padding=0;" src={img_rgba_encoded_url} width=20 height=20></img>
                        <span style="padding-left: 5px;">Interpolated surface</span>
                    </span>
                    """, unsafe_allow_html=True)
        for i, row in contours.iterrows():
            color_ = row.color_rgb
            color_str = ', '.join([str(x) for x in color_])
            contour_ = int(row.contour)
            st.markdown(f"""
                        <span style="display: inline-flex; align-items: center;">
                            <span style="background-color: rgba({color_str}, 1.0); border: 2px solid white, border-radius: 75%; display: inline-block; width: 20px; height: 4px"></span>
                            <span style="padding-left: 5px;">{contour_} m contour</span>
                        </span>
                        """, unsafe_allow_html=True)

# Explanations
with tab2:
    st.markdown(
        """
        ## About Accessibility and Isochrones
        
        This app creates walking distance `isochrones` for anywhere in the world that is already mapped in OpenStreetMap. Isochrones are lines of equal travel time or distance, an Accessibility concept used in Transportaiton and Urban Planning. The objective of the app is to allow the user to play with common algorithms and methodologies used in creating isochrones, to tweak their parameters and see how the result is affected. The isochrone calculation methods available in the app are:
        
        #### Computational Geometry methods (vector)
        - Convex Hull
        - Concave Hull
        - Link Offset
        
        #### Grid/Spatial Interpolation methods (raster)
        - Inverse Distance Weighting (IDW)
        - Triangular Irregular Network (TIN) / Linear
        
        In every case, the result is compared to the theoretical accessibility surface which is just a  buffer of `x` meters applied to the source where `x` is the maximum target distance. If you are unfamiliar with the the concept of isochrones or want more information, I have written a [full tutorial article here](https://medium.com/@arthur.dolmajian/creating-isochrones-what-is-the-optimal-way-dfc77a2ca13a).
        
        ## Controls
        
        The app is controlled with the sidebar and the results are shown on the right. Each map can be viewed full-screen by clicking the toggle <span style="margin: 0 7px 0 7px; padding: 5px; background-color: #34282C;"><svg viewBox="0 0 8 8" aria-hidden="true" focusable="false" fill="currentColor" xmlns="http://www.w3.org/2000/svg" color="inherit" class="e1fb0mya1 css-1pxazr7 ex0cdmw0"><path d="M0 0v4l1.5-1.5L3 4l1-1-1.5-1.5L4 0H0zm5 4L4 5l1.5 1.5L4 8h4V4L6.5 5.5 5 4z"></path></svg></span> to the top-right of a given map. The legend shows detailed information about the layers and the areas of each shape for easy comparison.
        There are two main controls.
        ### Location Controls
        Pan the map and position your desired location under the crosshair <span style="color: rgb(255, 75, 75); margin: 0 7px 0 7px;">⌖</span>. Once you have your location, click the `Set location` button to launch the calculation. *Panning the map will not automatically relaunch the calculation, only clicking the button will* (which is a good thing - thank you Streamlit forms!).
        ### Shape Approximation Controls
        For each method and algorithm, the user can tweak the settings and see the results in realtime. *Everytime a setting is changed, all of the calculations are relaunched.*
        
        ## Packages Used and Limitations
        
        The app works by downloading the walking network from [OpenStreetMap](https://www.openstreetmap.org/) using the [OSMnx](https://osmnx.readthedocs.io/en/stable/) package which in turn uses the [NetworkX](https://networkx.org/documentation/stable/index.html) and [GeoNetworkX](https://geonetworkx.readthedocs.io/en/latest/) packages for network calculations. Geospatial data manipulation is done using [GeoPandas](https://geopandas.org/en/stable/) and [Rasterio](https://rasterio.readthedocs.io/en/stable/) (for vector and raster data respectively) and [Pillow](https://pillow.readthedocs.io/en/stable/) is used for image format manipulation. [SciPy](https://scipy.org/) is used for the spatial/grid interpolation (to avoid using GDAL which would have been easier but installing GDAL is really a hit or miss each time, and I wanted to keep the package management simple for the sake of hosting on Streamlit). User interaction and visualization on web maps are achieved using [Folium](https://python-visualization.github.io/folium/) and [PyDeck](https://deckgl.readthedocs.io/en/latest/).
        
        The app attempts to calculate the extended ego graph first, but in some cases, OSM can have geometries that might not be interpreted correctly by Shapely and cause errors. When this happens, the app defaults to the standard ego graph.
        To keep things simple, the user chosen location is not merged to the graph and the cost of entry is not considered. Instead, the neareast node to the chosen location is used as the starting point of the isochrone but GeoNetworkX provides utility functions to accomplish just that and I might add it in the future.
        
        An app by [themapguy](https://arthurdolmajian.com/) | GitHub repo [github.com/adolmajian/isochrone-app](https://github.com/adolmajian/isochrone-app)
        """, unsafe_allow_html=True
    )
