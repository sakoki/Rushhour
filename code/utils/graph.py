import networkx as nx


def create_graph(gis_df):
    """Accept a GIS DataFrame and returns network graph

    Example structure of DataFrame:
    +-----------+-----------------------+
    | unique_ID | geometry              |
    +-----------+-----------------------+
    | 12345     | POLYGON((-122.23...)) |
    +-----------+-----------------------+
    | ...       | ...                   |
    +-----------+-----------------------+

    Where:
    unique_ID --> unique identifier for GIS structure
    geometry --> GIS POLYGON geometric structure

    Each entry (row) in the DataFrame is initialized as a node.
    For each POLYGON, if any points touch another POLYGON, an
    edge will be created between the nodes.

    The unique_ID and GIS POLYGON structure will be stored as
    node attributes.

    :param DataFrame gis_df: table of GIS POLYGON structures
    :return: network graph
    :rtype: networkx.classes.graph.Graph
    """

    G = nx.Graph()
    i = 1

    for shp in gis_df.itertuples():
        G.add_node(i, polygon=shp[2], geoid=shp[1])
        i += 1

    for n in G.nodes(data=True):
        state = n[1]['polygon']
        for o in G.nodes(data=True):
            other = o[1]['polygon']
            if state != other and state.touches(other):
                G.add_edge(n[0], o[0])

    return G


def n_nearest_neighbors(G,node_id):
    """Accepts a graph and finds the nearest neighbors of a node containing a particular attribute

    For this function, the graph is a representation of the GIS census zones.
    The node of the graph should have the geoid attribute, which is the unique
    identifier of the census region.

    :param graph: graph representation of census zones
    :type graph: networkx.classes.graph.Graph
    :param str attr: unique attribute of the node
    :return: list of nearest neighbor node id's
    :rtype: list of int
    """

    # get the node number with the corresponding attribute
    node = [node for node, attr in G.nodes(data=True) if attr['geoid'] == node_id]
    assert len(node) == 1, "More than one node have the specified geoid"
    nearest_nodes = list(G.neighbors(node[0]))
    neighbors = [G.node[i]['geoid'] for i in nearest_nodes]
    return neighbors