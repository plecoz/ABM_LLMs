import osmnx as ox
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