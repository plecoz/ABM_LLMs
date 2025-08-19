import osmnx as ox
import json
import matplotlib.pyplot as plt

class BaseMap:
    def __init__(self, graph):
        """Initialize the static street map."""
        self.fig, self.ax = ox.plot_graph(
            graph,
            show=False,
            close=False,
            bgcolor="#111111",  # Dark mode
            node_size=0,
            edge_color="lightgray",
            edge_linewidth=0.7
        )
        self.ax.set_title("Macau 15-Minute City", fontsize=16, color="white")
        self.ax.set_facecolor("#111111")
    
    def clear_artists(self):
        """Clear all dynamic elements."""
        for artist in self.ax.lines + self.ax.collections:
            #if artist not in self.ax._edges:
            if artist.axes is not None:  # Preserve base map edges
                artist.remove()


def plot_healthcare_access_from_json(json_paths, title: str = "Healthcare Accessibility"):
    """
    Plot static map of home nodes colored by healthcare access across one or more JSON files
    produced by OutputController.save_healthcare_access_points.

    Green: success True in any run
    Red: success explicitly False in all runs
    Gray: unknown (no data)
    """
    # Aggregate by (home_node, x, y)
    node_map = {}
    for path in (json_paths if isinstance(json_paths, (list, tuple)) else [json_paths]):
        with open(path, 'r') as f:
            payload = json.load(f)
        for rec in payload.get('points', []):
            node = rec.get('home_node')
            x, y = rec.get('x'), rec.get('y')
            succ = rec.get('success')
            if node is None or x is None or y is None:
                continue
            key = (node, x, y)
            if key not in node_map:
                node_map[key] = {'any_success': False, 'has_data': False}
            if succ is not None:
                node_map[key]['has_data'] = True
            if succ is True:
                node_map[key]['any_success'] = True

    xs_green, ys_green = [], []
    xs_red, ys_red = [], []
    xs_gray, ys_gray = [], []

    for (node, x, y), agg in node_map.items():
        if agg['any_success']:
            xs_green.append(x); ys_green.append(y)
        else:
            if agg['has_data']:
                xs_red.append(x); ys_red.append(y)
            else:
                xs_gray.append(x); ys_gray.append(y)

    plt.figure(figsize=(10, 8))
    if xs_gray:
        plt.scatter(xs_gray, ys_gray, s=8, c='#aaaaaa', label='Unknown (no data)')
    if xs_red:
        plt.scatter(xs_red, ys_red, s=10, c='red', label='No healthcare access')
    if xs_green:
        plt.scatter(xs_green, ys_green, s=10, c='green', label='Healthcare access')

    plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()