import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 180  # Increase timeout to 3 minutes

# Function to safely retrieve POIs and handle empty results
def get_pois(place, tags, poi_type, pois_dict):
    try:
        gdf = ox.features_from_place(place, tags)
        if len(gdf) > 0:
            pois_dict[poi_type] = gdf
            print(f"- Found {len(gdf)} {poi_type}")
        else:
            print(f"- No {poi_type} found")
            # Create empty GeoDataFrame with the same structure
            pois_dict[poi_type] = gpd.GeoDataFrame(
                data=[], 
                columns=['geometry'], 
                geometry='geometry',
                crs="EPSG:4326"
            )
    except Exception as e:
        print(f"- Error fetching {poi_type}: {e}")
        # Create empty GeoDataFrame
        pois_dict[poi_type] = gpd.GeoDataFrame(
            data=[], 
            columns=['geometry'], 
            geometry='geometry',
            crs="EPSG:4326"
        )

# Daily POIs class to store all daily service POIs
class DailyPOIs:
    def __init__(self, place):
        self.place = place
        self.pois = {}
        self.colors = {
            'groceries': 'red',
            'restaurants': 'green',
            'banks': 'blue',
            'hairdressers': 'purple',
            'post_offices': 'orange',
            'laundries': 'yellow'
        }
        
    def fetch_all(self):
        print("\nDownloading daily points of interest...")
        tags_groceries = {'shop': ['supermarket', 'grocery']}
        tags_banks = {'amenity': 'bank'}
        tags_restaurants = {'amenity': 'restaurant'}
        tags_barber_shops = {'shop': 'hairdresser'}
        tags_post_offices = {'amenity': 'post_office'}
        tags_laundries = {'amenity': 'laundry', 'shop': 'laundry'}
        
        # Get all daily POIs with error handling
        get_pois(self.place, tags_groceries, 'groceries', self.pois)
        get_pois(self.place, tags_restaurants, 'restaurants', self.pois)
        get_pois(self.place, tags_banks, 'banks', self.pois)
        get_pois(self.place, tags_barber_shops, 'hairdressers', self.pois)
        get_pois(self.place, tags_post_offices, 'post_offices', self.pois)
        get_pois(self.place, tags_laundries, 'laundries', self.pois)
    
    def plot(self, ax):
        for poi_type, color in self.colors.items():
            if poi_type in self.pois and len(self.pois[poi_type]) > 0:
                # Get centroids if there are any POIs
                centroids = self.pois[poi_type].geometry.centroid
                centroids.plot(ax=ax, color=color, markersize=50, alpha=0.7, label=poi_type.capitalize())

# Healthcare POIs class to store all healthcare related POIs
class HealthcarePOIs:
    def __init__(self, place):
        self.place = place
        self.pois = {}
        self.colors = {
            'hospitals': 'darkred',
            'clinics': 'salmon',
            'pharmacies': 'lightgreen',
            'health_centers': 'teal'
        }
        
    def fetch_all(self):
        print("\nDownloading healthcare points of interest...")
        tags_hospitals = {'amenity': 'hospital'}
        tags_clinics = {'amenity': 'clinic', 'healthcare': 'clinic'}
        tags_pharmacies = {'amenity': 'pharmacy'}
        tags_health_centers = {'healthcare': 'centre'}
        
        # Get all healthcare POIs with error handling
        get_pois(self.place, tags_hospitals, 'hospitals', self.pois)
        get_pois(self.place, tags_clinics, 'clinics', self.pois)
        get_pois(self.place, tags_pharmacies, 'pharmacies', self.pois)
        get_pois(self.place, tags_health_centers, 'health_centers', self.pois)
    
    def plot(self, ax):
        for poi_type, color in self.colors.items():
            if poi_type in self.pois and len(self.pois[poi_type]) > 0:
                # Get centroids if there are any POIs
                centroids = self.pois[poi_type].geometry.centroid
                centroids.plot(ax=ax, color=color, markersize=50, alpha=0.7, label=poi_type.capitalize().replace('_', ' '))

# Education POIs class to store all education related POIs
class EducationPOIs:
    def __init__(self, place):
        self.place = place
        self.pois = {}
        self.colors = {
            'kindergartens': 'pink',
            'primary_schools': 'orange',
            'secondary_schools': 'purple'
        }
        
    def fetch_all(self):
        print("\nDownloading education points of interest...")
        tags_kindergartens = {'amenity': 'kindergarten'}
        tags_primary_schools = {'amenity': 'school', 'isced:level': '1'}
        tags_secondary_schools = {'amenity': 'school', 'isced:level': ['2', '3']}
        
        # Try with more generic tag first for schools (OSM data quality varies)
        tags_schools = {'amenity': 'school'}
        
        # Get all education POIs with error handling
        get_pois(self.place, tags_kindergartens, 'kindergartens', self.pois)
        
        # For schools, we'll try the specific tags first, then fall back to generic if needed
        primary_count = 0
        secondary_count = 0
        
        try:
            primary_gdf = ox.features_from_place(self.place, tags_primary_schools)
            if len(primary_gdf) > 0:
                self.pois['primary_schools'] = primary_gdf
                primary_count = len(primary_gdf)
                print(f"- Found {primary_count} primary_schools")
            else:
                print("- No primary schools found with specific tags, using generic school tag")
                primary_count = 0
        except Exception:
            print("- Error fetching primary schools with specific tags, using generic school tag")
            primary_count = 0
            
        try:
            secondary_gdf = ox.features_from_place(self.place, tags_secondary_schools)
            if len(secondary_gdf) > 0:
                self.pois['secondary_schools'] = secondary_gdf
                secondary_count = len(secondary_gdf)
                print(f"- Found {secondary_count} secondary_schools")
            else:
                print("- No secondary schools found with specific tags, using generic school tag")
                secondary_count = 0
        except Exception:
            print("- Error fetching secondary schools with specific tags, using generic school tag")
            secondary_count = 0
        
        # If specific tags didn't work well, use generic school tag
        if primary_count == 0 or secondary_count == 0:
            try:
                schools_gdf = ox.features_from_place(self.place, tags_schools)
                if len(schools_gdf) > 0:
                    print(f"- Found {len(schools_gdf)} schools with generic tag")
                    
                    # If we didn't get primary schools, use half the generic schools
                    if primary_count == 0:
                        # Initialize empty GeoDataFrame for primary schools
                        if 'primary_schools' not in self.pois:
                            self.pois['primary_schools'] = gpd.GeoDataFrame(
                                data=[], 
                                columns=['geometry'], 
                                geometry='geometry',
                                crs="EPSG:4326"
                            )
                        # Add half of generic schools as primary
                        half_schools = len(schools_gdf) // 2
                        self.pois['primary_schools'] = schools_gdf.iloc[:half_schools]
                        print(f"- Assigned {half_schools} schools as primary schools")
                    
                    # If we didn't get secondary schools, use the other half
                    if secondary_count == 0:
                        # Initialize empty GeoDataFrame for secondary schools
                        if 'secondary_schools' not in self.pois:
                            self.pois['secondary_schools'] = gpd.GeoDataFrame(
                                data=[], 
                                columns=['geometry'], 
                                geometry='geometry',
                                crs="EPSG:4326"
                            )
                        # Add half of generic schools as secondary
                        half_schools = len(schools_gdf) // 2
                        self.pois['secondary_schools'] = schools_gdf.iloc[half_schools:]
                        print(f"- Assigned {len(schools_gdf) - half_schools} schools as secondary schools")
                else:
                    print("- No schools found with generic tag")
                    # Create empty GeoDataFrames
                    if 'primary_schools' not in self.pois:
                        self.pois['primary_schools'] = gpd.GeoDataFrame(
                            data=[], 
                            columns=['geometry'], 
                            geometry='geometry',
                            crs="EPSG:4326"
                        )
                    if 'secondary_schools' not in self.pois:
                        self.pois['secondary_schools'] = gpd.GeoDataFrame(
                            data=[], 
                            columns=['geometry'], 
                            geometry='geometry',
                            crs="EPSG:4326"
                        )
            except Exception as e:
                print(f"- Error fetching schools with generic tag: {e}")
                # Create empty GeoDataFrames if needed
                if 'primary_schools' not in self.pois:
                    self.pois['primary_schools'] = gpd.GeoDataFrame(
                        data=[], 
                        columns=['geometry'], 
                        geometry='geometry',
                        crs="EPSG:4326"
                    )
                if 'secondary_schools' not in self.pois:
                    self.pois['secondary_schools'] = gpd.GeoDataFrame(
                        data=[], 
                        columns=['geometry'], 
                        geometry='geometry',
                        crs="EPSG:4326"
                    )
    
    def plot(self, ax):
        for poi_type, color in self.colors.items():
            if poi_type in self.pois and len(self.pois[poi_type]) > 0:
                # Get centroids if there are any POIs
                centroids = self.pois[poi_type].geometry.centroid
                centroids.plot(ax=ax, color=color, markersize=50, alpha=0.7, label=poi_type.capitalize().replace('_', ' '))

# Entertainment POIs class to store all entertainment and leisure related POIs
class EntertainmentPOIs:
    def __init__(self, place):
        self.place = place
        self.pois = {}
        self.colors = {
            'parks': 'green',
            'squares': 'yellowgreen',
            'libraries': 'blue',
            'museums': 'purple',
            'art_galleries': 'magenta',
            'cultural_centers': 'indianred',
            'theaters': 'darkorange',
            'gyms': 'red',
            'stadiums': 'darkred'
        }
        
    def fetch_all(self):
        print("\nDownloading entertainment points of interest...")
        
        tags_parks = {'leisure': 'park'}
        tags_squares = {'place': 'square'}
        tags_libraries = {'amenity': 'library'}
        tags_museums = {'tourism': 'museum'}
        tags_art_galleries = {'tourism': 'gallery'}
        tags_cultural_centers = {'amenity': 'arts_centre'}
        tags_theaters = {'amenity': ['theatre', 'cinema']}
        tags_gyms = {'leisure': ['fitness_centre', 'sports_centre']}
        tags_stadiums = {'leisure': 'stadium'}
        
        # Get all entertainment POIs with error handling
        get_pois(self.place, tags_parks, 'parks', self.pois)
        get_pois(self.place, tags_squares, 'squares', self.pois)
        get_pois(self.place, tags_libraries, 'libraries', self.pois)
        get_pois(self.place, tags_museums, 'museums', self.pois)
        get_pois(self.place, tags_art_galleries, 'art_galleries', self.pois)
        get_pois(self.place, tags_cultural_centers, 'cultural_centers', self.pois)
        get_pois(self.place, tags_theaters, 'theaters', self.pois)
        get_pois(self.place, tags_gyms, 'gyms', self.pois)
        get_pois(self.place, tags_stadiums, 'stadiums', self.pois)
    
    def plot(self, ax):
        for poi_type, color in self.colors.items():
            if poi_type in self.pois and len(self.pois[poi_type]) > 0:
                # Get centroids if there are any POIs
                centroids = self.pois[poi_type].geometry.centroid
                centroids.plot(ax=ax, color=color, markersize=50, alpha=0.7, label=poi_type.capitalize().replace('_', ' '))

# Public Transportation POIs class to store public transportation related POIs
class PublicTransportationPOIs:
    def __init__(self, place):
        self.place = place
        self.pois = {}
        self.colors = {
            'bus_stops': 'blue'
        }
        
    def fetch_all(self):
        print("\nDownloading public transportation points of interest...")
        
        tags_bus_stops = {'highway': 'bus_stop'}
        
        # Get only bus stops with error handling
        get_pois(self.place, tags_bus_stops, 'bus_stops', self.pois)
    
    def plot(self, ax):
        for poi_type, color in self.colors.items():
            if poi_type in self.pois and len(self.pois[poi_type]) > 0:
                # Get centroids if there are any POIs
                centroids = self.pois[poi_type].geometry.centroid
                centroids.plot(ax=ax, color=color, markersize=50, alpha=0.7, label=poi_type.capitalize().replace('_', ' '))

try:
    # Download Macau street network
    print("Downloading 'Macau, China' street network...")
    graph = ox.graph_from_place(
        "Macau, China", 
        network_type="walk",  # Pedestrian paths only
        simplify=True        # Clean topological artifacts
    )
    
    print(f"Success! Loaded:")
    print(f"- Nodes (intersections): {len(graph.nodes())}")
    print(f"- Edges (streets): {len(graph.edges())}")
    print(f"- Total walkable length: {ox.stats.edge_length_total(graph)/1000:.1f} km")
    
    # Initialize POI classes
    daily_pois = DailyPOIs('Macau, China')
    healthcare_pois = HealthcarePOIs('Macau, China')
    education_pois = EducationPOIs('Macau, China')
    entertainment_pois = EntertainmentPOIs('Macau, China')
    transportation_pois = PublicTransportationPOIs('Macau, China')
    
    # Fetch POIs
    # daily_pois.fetch_all()  # Commented out to stop plotting daily POIs
    # healthcare_pois.fetch_all()  # Keep fetching but don't plot
    # education_pois.fetch_all()  # Commented out to stop plotting education POIs
    # entertainment_pois.fetch_all()  # Commented out to stop plotting entertainment POIs
    transportation_pois.fetch_all()  # Fetch and plot public transportation POIs
    
    # Create figure with single axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot street network first (background)
    ox.plot_graph(
        graph,
        bgcolor="#f5f5f5",  # Light gray background
        node_size=0,        # Hide nodes
        edge_color="#4682b4",  # Steel blue streets
        edge_linewidth=0.7,
        show=False,
        ax=ax
    )
    
    # Plot POIs
    # daily_pois.plot(ax)  # Commented out to stop plotting daily POIs
    # healthcare_pois.plot(ax)  # Commented out to stop plotting healthcare POIs
    # education_pois.plot(ax)  # Commented out to stop plotting education POIs
    # entertainment_pois.plot(ax)  # Commented out to stop plotting entertainment POIs
    transportation_pois.plot(ax)  # Plot public transportation POIs
    
    # Add title and legend
    ax.set_title("Macau Walkable Streets with Public Transportation", fontsize=16, pad=20)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('macau_public_transportation.png', dpi=300)
    plt.show()

except Exception as e:
    print(f"\n❌ Error: {e}")