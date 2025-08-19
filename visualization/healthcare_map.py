import os
import json
import pickle
from typing import List, Union, Dict, Any

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import geopandas as gpd


def _load_graph(graph_path: str) -> nx.Graph:
    # NetworkX 3.x may not expose read_gpickle at top-level
    try:
        from networkx.readwrite.gpickle import read_gpickle as nx_read_gpickle  # type: ignore
        return nx_read_gpickle(graph_path)
    except Exception:
        # Fallback: load via pickle directly
        with open(graph_path, "rb") as f:
            return pickle.load(f)


def _load_environment(env_path: str) -> Dict[str, Any]:
    if not env_path or not os.path.exists(env_path):
        return {
            "residential_buildings": gpd.GeoDataFrame(),
            "water_bodies": gpd.GeoDataFrame(),
            "cliffs": gpd.GeoDataFrame(),
            "forests": gpd.GeoDataFrame(),
        }
    with open(env_path, "rb") as f:
        data = pickle.load(f)
    # Ensure expected keys exist
    return {
        "residential_buildings": data.get("residential_buildings", gpd.GeoDataFrame()),
        "water_bodies": data.get("water_bodies", gpd.GeoDataFrame()),
        "cliffs": data.get("cliffs", gpd.GeoDataFrame()),
        "forests": data.get("forests", gpd.GeoDataFrame()),
    }


def _load_pois(pois_path: str) -> Dict[str, list]:
    if not pois_path or not os.path.exists(pois_path):
        return {}
    with open(pois_path, "rb") as f:
        return pickle.load(f)


def _load_parishes(parishes_path: str) -> gpd.GeoDataFrame:
    if not parishes_path or not os.path.exists(parishes_path):
        return gpd.GeoDataFrame()
    return gpd.read_file(parishes_path)


def _aggregate_health_points(json_paths: Union[str, List[str]]):
    paths = json_paths if isinstance(json_paths, (list, tuple)) else [json_paths]

    xs_green, ys_green = [], []
    xs_red, ys_red = [], []

    for path in paths:
        with open(path, "r") as f:
            payload = json.load(f)
        for rec in payload.get("points", []):
            x, y, succ = rec.get("x"), rec.get("y"), rec.get("success")
            if x is None or y is None:
                continue
            if succ is True:
                xs_green.append(x); ys_green.append(y)
            elif succ is False:
                xs_red.append(x); ys_red.append(y)
            # None = unknown; not plotted here to keep the map clean

    return (xs_green, ys_green, xs_red, ys_red)


def _collect_poi_coords(graph: nx.Graph, pois: Dict[str, list]) -> Dict[str, Dict[str, list]]:
    category_to_points: Dict[str, Dict[str, list]] = {}
    if not pois:
        return category_to_points

    for category, poi_list in pois.items():
        xs, ys = [], []
        for item in poi_list:
            if isinstance(item, tuple):
                node_id, _poi_type = item
            else:
                node_id = item
            if node_id in graph.nodes:
                node = graph.nodes[node_id]
                x, y = node.get("x"), node.get("y")
                if x is not None and y is not None:
                    xs.append(x); ys.append(y)
        if xs:
            category_to_points[category] = {"x": xs, "y": ys}
    return category_to_points


def plot_full_healthcare_map(
    healthcare_json: Union[str, List[str]],
    graph_path: str = "data/macau_shapefiles/macau_network.pkl",
    parishes_path: str = "data/macau_shapefiles/macau_new_districts.gpkg",
    environment_path: str = "data/macau_shapefiles/macau_environment.pkl",
    pois_path: str = "data/macau_shapefiles/macau_pois.pkl",
    out_path: str = None,
    title: str = "Macau Healthcare Accessibility"
):
    """
    Produce a static map with street network, parishes, buildings, POIs and
    healthcare access outcome points (green/red), optionally saving to file.
    """

    # Load data
    G = _load_graph(graph_path)
    env = _load_environment(environment_path)
    parishes_gdf = _load_parishes(parishes_path)
    pois = _load_pois(pois_path)
    xs_g, ys_g, xs_r, ys_r = _aggregate_health_points(healthcare_json)

    # Base map from street network
    fig, ax = ox.plot_graph(
        G,
        show=False,
        close=False,
        bgcolor="#111111",
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.6
    )
    ax.set_facecolor("#111111")
    ax.set_title(title, fontsize=14, color="white")

    # Overlay parishes (boundaries only, thin lines)
    if not parishes_gdf.empty:
        try:
            parishes_gdf.boundary.plot(ax=ax, color="#FFD166", linewidth=0.8, alpha=0.8)
        except Exception:
            pass

    # Overlay buildings (light fill, subtle)
    bld = env.get("residential_buildings", gpd.GeoDataFrame())
    if bld is not None and not bld.empty:
        try:
            bld.plot(ax=ax, facecolor="#999999", edgecolor="#666666", linewidth=0.1, alpha=0.25)
        except Exception:
            pass

    # Optional: natural features (very subtle)
    forests = env.get("forests", gpd.GeoDataFrame())
    if forests is not None and not forests.empty:
        try:
            forests.plot(ax=ax, facecolor="#2E8B57", edgecolor="#2E8B57", linewidth=0.1, alpha=0.15)
        except Exception:
            pass

    water = env.get("water_bodies", gpd.GeoDataFrame())
    if water is not None and not water.empty:
        try:
            water.plot(ax=ax, facecolor="#3A86FF", edgecolor="#3A86FF", linewidth=0.2, alpha=0.25)
        except Exception:
            pass

    # Plot POIs (healthcare highlighted, others dim)
    poi_points = _collect_poi_coords(G, pois)
    if poi_points:
        for category, pts in poi_points.items():
            xs, ys = pts["x"], pts["y"]
            if not xs:
                continue
            if category.lower() == "healthcare":
                ax.scatter(xs, ys, s=8, c="#4CC9F0", marker="x", label="Healthcare POIs", alpha=0.9)
            else:
                ax.scatter(xs, ys, s=4, c="#888888", marker=".", alpha=0.5)

    # Overlay healthcare access outcomes
    if xs_r:
        ax.scatter(xs_r, ys_r, s=14, c="red", label="No healthcare access")
    if xs_g:
        ax.scatter(xs_g, ys_g, s=14, c="lime", label="Healthcare access")

    # Legend & layout
    try:
        ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="white")
    except Exception:
        pass
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=220)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Macau healthcare accessibility map")
    parser.add_argument("--hc-json", nargs="+", required=True, help="Healthcare points JSON(s)")
    parser.add_argument("--graph-path", default="data/macau_shapefiles/macau_network.pkl")
    parser.add_argument("--parishes-path", default="data/macau_shapefiles/macau_new_districts.gpkg")
    parser.add_argument("--environment-path", default="data/macau_shapefiles/macau_environment.pkl")
    parser.add_argument("--pois-path", default="data/macau_shapefiles/macau_pois.pkl")
    parser.add_argument("--out", help="Save to file instead of showing (e.g., reports/healthcare_map.png)")
    args = parser.parse_args()

    plot_full_healthcare_map(
        healthcare_json=args.hc_json,
        graph_path=args.graph_path,
        parishes_path=args.parishes_path,
        environment_path=args.environment_path,
        pois_path=args.pois_path,
        out_path=args.out,
    )


