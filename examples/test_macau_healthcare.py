#!/usr/bin/env python3
"""
Healthcare-focused visualization for ABM design in Macau.
This file creates visualizations specifically for healthcare policy making in ABM,
focusing on the parishes of Taipa and Coloane.
"""

import os
import pickle
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

# Configure OSMnx
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.timeout = 180

class HealthcareFacilities:
    """Class to handle comprehensive healthcare facility data from OSM."""
    
    def __init__(self, place_name="Macau, China"):
        self.place_name = place_name
        self.facilities = {}
        self.colors = {
            'hospitals': '#8B0000',      # Dark red
            'clinics': '#DC143C',        # Crimson  
            'pharmacies': '#32CD32',     # Lime green
            'dentists': '#4169E1',       # Royal blue
            'health_centers': '#008B8B',  # Dark cyan
            'nursing_homes': '#9932CC',   # Dark orchid
            'laboratories': '#FF8C00',    # Dark orange
            'medical_supply': '#2E8B57',  # Sea green
            'emergency_services': '#FF0000', # Red
            'rehabilitation': '#4682B4'   # Steel blue
        }
    
    def fetch_all_healthcare_facilities(self):
        """Fetch comprehensive healthcare facilities with detailed metadata."""
        print(f"\nFetching healthcare facilities for {self.place_name}...")
        
        # Define comprehensive healthcare tags
        healthcare_tags = {
            'hospitals': {
                'amenity': 'hospital',
                'healthcare': 'hospital'
            },
            'clinics': {
                'amenity': 'clinic',
                'healthcare': ['clinic', 'doctor', 'general_practitioner']
            },
            'pharmacies': {
                'amenity': 'pharmacy',
                'shop': 'pharmacy'
            },
            'dentists': {
                'amenity': 'dentist',
                'healthcare': 'dentist'
            },
            'health_centers': {
                'healthcare': 'centre',
                'amenity': 'health_centre'
            },
            'nursing_homes': {
                'amenity': 'nursing_home',
                'healthcare': 'nursing_home'
            },
            'laboratories': {
                'healthcare': 'laboratory',
                'amenity': 'laboratory'
            },
            'medical_supply': {
                'shop': 'medical_supply',
                'healthcare': 'medical_supply'
            },
            'emergency_services': {
                'emergency': ['ambulance_station', 'emergency_ward'],
                'healthcare': 'emergency'
            },
            'rehabilitation': {
                'healthcare': ['rehabilitation', 'physiotherapy'],
                'amenity': 'physiotherapy'
            }
        }
        
        for facility_type, tags in healthcare_tags.items():
            self._fetch_facility_type(facility_type, tags)
    
    def _fetch_facility_type(self, facility_type, tags):
        """Fetch a specific type of healthcare facility."""
        try:
            print(f"- Fetching {facility_type}...")
            gdf = ox.features_from_place(self.place_name, tags)
            
            if len(gdf) > 0:
                # Extract useful metadata
                metadata_columns = [
                    'name', 'opening_hours', 'phone', 'website', 'operator',
                    'healthcare:speciality', 'beds', 'emergency', 'wheelchair',
                    'addr:street', 'addr:housenumber', 'addr:postcode',
                    'operator:type'  # To determine public/private status
                ]
                
                # Keep only columns that exist in the data
                available_columns = [col for col in metadata_columns if col in gdf.columns]
                
                # Create a clean GeoDataFrame with metadata
                clean_gdf = gdf[['geometry'] + available_columns].copy()
                
                # Add facility type
                clean_gdf['facility_type'] = facility_type
                
                # Try to determine public/private status
                clean_gdf['status'] = self._determine_status(clean_gdf)
                
                self.facilities[facility_type] = clean_gdf
                print(f"  Found {len(clean_gdf)} {facility_type}")
                
                # Print detailed metadata info
                self._print_metadata_summary(clean_gdf, facility_type)
                    
            else:
                print(f"  No {facility_type} found")
                self.facilities[facility_type] = gpd.GeoDataFrame(
                    columns=['geometry', 'facility_type', 'status'],
                    geometry='geometry',
                    crs="EPSG:4326"
                )
                
        except Exception as e:
            print(f"  Error fetching {facility_type}: {e}")
            self.facilities[facility_type] = gpd.GeoDataFrame(
                columns=['geometry', 'facility_type', 'status'],
                geometry='geometry',
                crs="EPSG:4326"
            )
    
    def _determine_status(self, gdf):
        """Determine public/private status based on operator information."""
        status = []
        for _, row in gdf.iterrows():
            operator = row.get('operator', '')
            operator_type = row.get('operator:type', '')
            
            if pd.isna(operator) and pd.isna(operator_type):
                status.append('unknown')
            elif any(keyword in str(operator).lower() for keyword in ['government', 'public', 'municipal', 'state']):
                status.append('public')
            elif any(keyword in str(operator_type).lower() for keyword in ['government', 'public']):
                status.append('public')
            elif any(keyword in str(operator).lower() for keyword in ['private', 'ltd', 'company', 'corp']):
                status.append('private')
            elif str(operator_type).lower() == 'private':
                status.append('private')
            else:
                status.append('unknown')
        
        return status
    
    def _print_metadata_summary(self, gdf, facility_type):
        """Print detailed metadata summary for facilities."""
        print(f"    Metadata available for {facility_type}:")
        
        metadata_fields = {
            'name': 'Named facilities',
            'opening_hours': 'Opening hours',
            'phone': 'Phone numbers',
            'website': 'Websites',
            'operator': 'Operator info',
            'healthcare:speciality': 'Medical specialties',
            'beds': 'Bed capacity',
            'emergency': 'Emergency services',
            'wheelchair': 'Wheelchair accessibility',
            'addr:street': 'Street addresses',
            'addr:housenumber': 'House numbers',
            'addr:postcode': 'Postal codes',
            'operator:type': 'Operator type'
        }
        
        for field, description in metadata_fields.items():
            if field in gdf.columns:
                count = gdf[field].notna().sum()
                percentage = (count / len(gdf)) * 100
                print(f"      - {description}: {count}/{len(gdf)} ({percentage:.1f}%)")
        
        # Print status distribution
        if 'status' in gdf.columns:
            status_counts = gdf['status'].value_counts()
            print(f"      - Status distribution:")
            for status, count in status_counts.items():
                percentage = (count / len(gdf)) * 100
                print(f"        * {status.title()}: {count} ({percentage:.1f}%)")
    
    def get_all_facilities_combined(self):
        """Combine all healthcare facilities into a single GeoDataFrame."""
        all_facilities = []
        for facility_type, gdf in self.facilities.items():
            if len(gdf) > 0:
                all_facilities.append(gdf)
        
        if all_facilities:
            return pd.concat(all_facilities, ignore_index=True)
        else:
            return gpd.GeoDataFrame(
                columns=['geometry', 'facility_type', 'status'],
                geometry='geometry',
                crs="EPSG:4326"
            )
    
    def plot_facilities(self, ax, parishes_gdf=None):
        """Plot all healthcare facilities with legend."""
        legend_elements = []
        
        for facility_type, color in self.colors.items():
            if facility_type in self.facilities and len(self.facilities[facility_type]) > 0:
                gdf = self.facilities[facility_type]
                
                # Plot facility locations
                centroids = gdf.geometry.centroid
                centroids.plot(ax=ax, color=color, markersize=80, alpha=0.8, 
                             edgecolor='black', linewidth=0.5)
                
                # Add to legend
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                label=f"{facility_type.replace('_', ' ').title()} ({len(gdf)})",
                                                markerfacecolor=color, markersize=8))
        
        return legend_elements

class ResidentialZoning:
    """Class to handle residential building zoning information."""
    
    def __init__(self, place_name="Macau, China"):
        self.place_name = place_name
        self.residential_data = {}
    
    def fetch_residential_zoning(self):
        """Fetch residential buildings and zoning information."""
        print(f"\nFetching residential zoning data for {self.place_name}...")
        
        # Comprehensive residential tags
        residential_tags = {
            'residential_buildings': {
                'building': ['apartments', 'house', 'residential', 'detached', 
                           'semidetached_house', 'terrace', 'dormitory', 'bungalow']
            },
            'residential_landuse': {
                'landuse': 'residential'
            },
            'housing_estates': {
                'place': 'neighbourhood',
                'residential': 'housing_estate'
            }
        }
        
        for zone_type, tags in residential_tags.items():
            self._fetch_residential_type(zone_type, tags)
    
    def _fetch_residential_type(self, zone_type, tags):
        """Fetch a specific type of residential zoning."""
        try:
            print(f"- Fetching {zone_type}...")
            gdf = ox.features_from_place(self.place_name, tags)
            
            if len(gdf) > 0:
                # Keep relevant columns
                metadata_columns = [
                    'name', 'addr:street', 'addr:housenumber', 'building:levels',
                    'building:use', 'residential', 'landuse'
                ]
                available_columns = [col for col in metadata_columns if col in gdf.columns]
                clean_gdf = gdf[['geometry'] + available_columns].copy()
                clean_gdf['zone_type'] = zone_type
                
                self.residential_data[zone_type] = clean_gdf
                print(f"  Found {len(clean_gdf)} {zone_type.replace('_', ' ')}")
            else:
                print(f"  No {zone_type.replace('_', ' ')} found")
                self.residential_data[zone_type] = gpd.GeoDataFrame(
                    columns=['geometry', 'zone_type'],
                    geometry='geometry',
                    crs="EPSG:4326"
                )
        except Exception as e:
            print(f"  Error fetching {zone_type}: {e}")
            self.residential_data[zone_type] = gpd.GeoDataFrame(
                columns=['geometry', 'zone_type'],
                geometry='geometry',
                crs="EPSG:4326"
            )
    
    def plot_residential_zones(self, ax):
        """Plot residential zoning with different styles."""
        colors = {
            'residential_buildings': '#FFB6C1',  # Light pink
            'residential_landuse': '#FFA07A',    # Light salmon
            'housing_estates': '#DDA0DD'         # Plum
        }
        
        legend_elements = []
        
        for zone_type, gdf in self.residential_data.items():
            if len(gdf) > 0:
                color = colors.get(zone_type, '#CCCCCC')
                
                if zone_type == 'residential_buildings':
                    # Plot building outlines
                    gdf.plot(ax=ax, color=color, alpha=0.6, edgecolor='darkred', linewidth=0.5)
                else:
                    # Plot area zones
                    gdf.plot(ax=ax, color=color, alpha=0.4, edgecolor='black', linewidth=0.8)
                
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6,
                                                   label=f"{zone_type.replace('_', ' ').title()} ({len(gdf)})"))
        
        return legend_elements

def load_environment_data(file_path):
    """Load environment data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading environment data: {e}")
        return None

def filter_by_parishes(gdf, parishes_gdf, target_parishes):
    """Filter GeoDataFrame to only include features in target parishes."""
    if len(gdf) == 0:
        return gdf
    
    # Get target parish geometries
    target_geoms = parishes_gdf[parishes_gdf['name'].isin(target_parishes)]['geometry']
    if len(target_geoms) == 0:
        print(f"Warning: No parishes found with names {target_parishes}")
        return gdf
    
    # Create union of target parish geometries
    target_union = target_geoms.unary_union
    
    # Filter features that intersect with target parishes
    mask = gdf.geometry.intersects(target_union)
    return gdf[mask].copy()

def create_healthcare_abm_visualization():
    """Create comprehensive healthcare ABM visualization for Macau."""
    
    # File paths
    data_dir = "data/macau_shapefiles"
    environment_file = os.path.join(data_dir, "macau_environment.pkl")
    districts_file = os.path.join(data_dir, "macau_new_districts.gpkg")
    output_file = os.path.join(data_dir, "macau_healthcare.pkl")
    
    # Target parishes
    target_parishes = ["Taipa", "Coloane"]
    
    print("=== Healthcare ABM Visualization for Macau ===")
    print(f"Target parishes: {', '.join(target_parishes)}")
    
    # Load districts/parishes data
    print(f"\nLoading parish boundaries from {districts_file}...")
    try:
        parishes_gdf = gpd.read_file(districts_file)
        print(f"Loaded {len(parishes_gdf)} parishes")
        print(f"Available parishes: {', '.join(parishes_gdf['name'].tolist())}")
        
        # Filter to target parishes
        target_parishes_gdf = parishes_gdf[parishes_gdf['name'].isin(target_parishes)]
        if len(target_parishes_gdf) == 0:
            print("Error: Target parishes not found in the data")
            return
        print(f"Found {len(target_parishes_gdf)} target parishes")
        
    except Exception as e:
        print(f"Error loading parish data: {e}")
        return
    
    # Load environment data
    print(f"\nLoading environment data from {environment_file}...")
    env_data = load_environment_data(environment_file)
    if env_data is None:
        print("Error: Could not load environment data")
        return
    
    # Extract environment components and filter by parishes
    print("\nFiltering environment data to target parishes...")
    residential_buildings = filter_by_parishes(
        env_data.get('residential_buildings', gpd.GeoDataFrame()), 
        parishes_gdf, target_parishes
    )
    water_bodies = filter_by_parishes(
        env_data.get('water_bodies', gpd.GeoDataFrame()), 
        parishes_gdf, target_parishes
    )
    forests = filter_by_parishes(
        env_data.get('forests', gpd.GeoDataFrame()), 
        parishes_gdf, target_parishes
    )
    
    print(f"Filtered environment data:")
    print(f"- Residential buildings: {len(residential_buildings)}")
    print(f"- Water bodies: {len(water_bodies)}")
    print(f"- Forests/green areas: {len(forests)}")
    
    # Fetch healthcare facilities
    healthcare = HealthcareFacilities("Macau, China")
    healthcare.fetch_all_healthcare_facilities()
    
    # Filter healthcare facilities to target parishes
    print("\nFiltering healthcare facilities to target parishes...")
    for facility_type in healthcare.facilities:
        original_count = len(healthcare.facilities[facility_type])
        healthcare.facilities[facility_type] = filter_by_parishes(
            healthcare.facilities[facility_type],
            parishes_gdf, target_parishes
        )
        filtered_count = len(healthcare.facilities[facility_type])
        if original_count > 0:
            print(f"- {facility_type}: {filtered_count}/{original_count} facilities in target parishes")
    
    # Fetch residential zoning
    residential_zoning = ResidentialZoning("Macau, China")
    residential_zoning.fetch_residential_zoning()
    
    # Filter residential zoning to target parishes
    print("\nFiltering residential zoning to target parishes...")
    for zone_type in residential_zoning.residential_data:
        residential_zoning.residential_data[zone_type] = filter_by_parishes(
            residential_zoning.residential_data[zone_type],
            parishes_gdf, target_parishes
        )
    
    # Create visualization
    print("\nCreating healthcare ABM visualization...")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot parish boundaries
    target_parishes_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.8)
    
    # Plot environment components
    if len(water_bodies) > 0:
        water_bodies.plot(ax=ax, color='lightblue', alpha=0.7, label=f'Water Bodies ({len(water_bodies)})')
    
    if len(forests) > 0:
        forests.plot(ax=ax, color='lightgreen', alpha=0.5, label=f'Green Areas ({len(forests)})')
    
    # Plot residential zoning
    residential_legend = residential_zoning.plot_residential_zones(ax)
    
    # Plot healthcare facilities
    healthcare_legend = healthcare.plot_facilities(ax, target_parishes_gdf)
    
    # Set the plot extent to focus on target parishes with some padding
    bounds = target_parishes_gdf.total_bounds  # [minx, miny, maxx, maxy]
    padding = 0.01  # Add padding around the parishes (in degrees)
    ax.set_xlim(bounds[0] - padding, bounds[2] + padding)
    ax.set_ylim(bounds[1] - padding, bounds[3] + padding)
    
    # Customize plot
    ax.set_title(f'Healthcare ABM Environment - {", ".join(target_parishes)} Parishes, Macau', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Create comprehensive legend
    all_legend_elements = []
    
    # Add parish boundary to legend
    all_legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2, 
                                        label='Parish Boundaries'))
    
    # Add environment elements
    if len(water_bodies) > 0:
        all_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7,
                                                label=f'Water Bodies ({len(water_bodies)})'))
    if len(forests) > 0:
        all_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.5,
                                                label=f'Green Areas ({len(forests)})'))
    
    # Add residential zoning
    all_legend_elements.extend(residential_legend)
    
    # Add healthcare facilities
    all_legend_elements.extend(healthcare_legend)
    
    # Place legend outside the plot
    ax.legend(handles=all_legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=10, title='Legend', title_fontsize=12)
    
    # Set equal aspect ratio and tight layout
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # Save the plot
    output_image = f"macau_healthcare_abm_{'-'.join(target_parishes).lower()}.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved as: {output_image}")
    
    # Print comprehensive metadata summary
    print("\n=== COMPREHENSIVE METADATA SUMMARY ===")
    total_facilities = 0
    for facility_type, gdf in healthcare.facilities.items():
        if len(gdf) > 0:
            total_facilities += len(gdf)
            print(f"\n{facility_type.replace('_', ' ').title()} ({len(gdf)} facilities):")
            healthcare._print_metadata_summary(gdf, facility_type)
    
    if total_facilities == 0:
        print("No healthcare facilities found in the target parishes.")
    else:
        print(f"\nTotal healthcare facilities in target parishes: {total_facilities}")
    
    # Save healthcare data to pickle file
    print(f"\nSaving healthcare data to {output_file}...")
    
    healthcare_data = {
        'parishes': target_parishes,
        'parish_boundaries': target_parishes_gdf,
        'healthcare_facilities': healthcare.facilities,
        'residential_zoning': residential_zoning.residential_data,
        'environment_data': {
            'residential_buildings': residential_buildings,
            'water_bodies': water_bodies,
            'forests': forests
        },
        'metadata': {
            'total_healthcare_facilities': sum(len(gdf) for gdf in healthcare.facilities.values()),
            'total_residential_buildings': len(residential_buildings),
            'facility_types': list(healthcare.facilities.keys()),
            'zoning_types': list(residential_zoning.residential_data.keys())
        }
    }
    
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(healthcare_data, f)
        print(f"Healthcare data successfully saved to {output_file}")
        
        # Print summary
        print("\n=== Healthcare ABM Data Summary ===")
        print(f"Parishes: {', '.join(target_parishes)}")
        print(f"Total healthcare facilities: {healthcare_data['metadata']['total_healthcare_facilities']}")
        for facility_type, gdf in healthcare.facilities.items():
            if len(gdf) > 0:
                print(f"  - {facility_type.replace('_', ' ').title()}: {len(gdf)}")
        print(f"Total residential buildings: {len(residential_buildings)}")
        print(f"Water bodies: {len(water_bodies)}")
        print(f"Green areas: {len(forests)}")
        
    except Exception as e:
        print(f"Error saving healthcare data: {e}")
    
    plt.show()
    
    return healthcare_data

if __name__ == "__main__":
    create_healthcare_abm_visualization() 