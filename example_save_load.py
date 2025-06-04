#!/usr/bin/env python3
"""
Example script demonstrating how to save and load network and POIs for faster testing.

This script shows three scenarios:
1. First run: Load from OSM and save to files
2. Subsequent runs: Load from saved files (much faster)
3. Mixed approach: Load network from file, fetch fresh POIs
"""

import os
import sys
from main import run_simulation

def main():
    print("=== 15-Minute City Simulation - Save/Load Example ===\n")
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Define file paths
    network_file = os.path.join(data_dir, "macau_network.pkl")
    pois_file = os.path.join(data_dir, "macau_pois.pkl")
    
    print("Choose an option:")
    print("1. First run - Load from OSM and save to files (slower, but saves for future)")
    print("2. Fast run - Load from saved files (much faster)")
    print("3. Mixed - Load network from file, fetch fresh POIs")
    print("4. Clean run - Delete saved files and start fresh")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n=== FIRST RUN: Loading from OSM and saving to files ===")
        print("This will take longer but will save files for faster future runs.")
        
        run_simulation(
            num_residents=50,  # Smaller number for testing
            steps=120,         # 2 hours for testing
            save_network=network_file,  # Save network after loading
            save_pois=pois_file,       # Save POIs after fetching
            movement_behavior='random',
            seed=42  # Fixed seed for reproducible results
        )
        
        print(f"\nFiles saved:")
        print(f"- Network: {network_file}")
        print(f"- POIs: {pois_file}")
        print("Next time, use option 2 for much faster loading!")
        
    elif choice == "2":
        print("\n=== FAST RUN: Loading from saved files ===")
        
        # Check if files exist
        if not os.path.exists(network_file):
            print(f"Error: Network file not found: {network_file}")
            print("Please run option 1 first to create the saved files.")
            return
            
        if not os.path.exists(pois_file):
            print(f"Error: POIs file not found: {pois_file}")
            print("Please run option 1 first to create the saved files.")
            return
        
        print("Loading from saved files - this should be much faster!")
        
        run_simulation(
            num_residents=50,
            steps=120,
            load_network=network_file,  # Load network from file
            load_pois=pois_file,       # Load POIs from file
            movement_behavior='random',
            seed=42  # Use same seed for reproducible comparison
        )
        
    elif choice == "3":
        print("\n=== MIXED RUN: Load network from file, fetch fresh POIs ===")
        
        if not os.path.exists(network_file):
            print(f"Error: Network file not found: {network_file}")
            print("Please run option 1 first to create the network file.")
            return
        
        print("Loading network from file, fetching fresh POIs from OSM...")
        
        run_simulation(
            num_residents=50,
            steps=120,
            load_network=network_file,  # Load network from file
            # No load_pois - will fetch from OSM
            save_pois=pois_file,       # Save the fresh POIs
            movement_behavior='random',
            seed=42  # Use same seed for reproducible comparison
        )
        
    elif choice == "4":
        print("\n=== CLEAN RUN: Deleting saved files ===")
        
        files_deleted = []
        if os.path.exists(network_file):
            os.remove(network_file)
            files_deleted.append(network_file)
            
        if os.path.exists(pois_file):
            os.remove(pois_file)
            files_deleted.append(pois_file)
        
        if files_deleted:
            print(f"Deleted files: {', '.join(files_deleted)}")
        else:
            print("No saved files found to delete.")
            
        print("Now running fresh simulation from OSM...")
        
        run_simulation(
            num_residents=50,
            steps=120,
            movement_behavior='random',
            seed=42  # Use same seed for reproducible comparison
        )
        
    else:
        print("Invalid choice. Please run the script again and choose 1-4.")

if __name__ == "__main__":
    main() 