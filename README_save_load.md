# Save/Load Functionality for Faster Testing

The simulation now supports saving and loading the city network and POIs to/from local files, which significantly speeds up testing by avoiding repeated OpenStreetMap queries.

## Why Use Save/Load?

- **Faster Testing**: Loading from files is much faster than querying OpenStreetMap
- **Offline Development**: Work without internet connection once files are saved
- **Consistent Data**: Use the same network/POIs across multiple test runs
- **Reduced API Load**: Fewer requests to OpenStreetMap servers

## Command Line Arguments

### Save Options
- `--save-network PATH`: Save the city network after loading from OSM
- `--save-pois PATH`: Save the POIs after fetching from OSM

### Load Options
- `--load-network PATH`: Load the city network from file instead of OSM
- `--load-pois PATH`: Load the POIs from file instead of OSM

### Reproducibility
- `--seed INT`: Set random seed for reproducible results (e.g., `--seed 42`)

## Usage Examples

### 1. First Run - Save Everything with Fixed Seed
```bash
python main.py --save-network data/macau_network.pkl --save-pois data/macau_pois.pkl --residents 50 --steps 120 --seed 42
```

### 2. Fast Subsequent Runs - Load Everything with Same Seed
```bash
python main.py --load-network data/macau_network.pkl --load-pois data/macau_pois.pkl --residents 50 --steps 120 --seed 42
```

### 3. Mixed Approach - Load Network, Fresh POIs
```bash
python main.py --load-network data/macau_network.pkl --save-pois data/macau_pois_new.pkl --residents 50 --steps 120 --seed 42
```

### 4. Save Only Network (POIs from OSM)
```bash
python main.py --save-network data/macau_network.pkl --residents 50 --steps 120 --seed 42
```

## Interactive Example Script

Run the example script for a guided experience:

```bash
python example_save_load.py
```

This script provides four options and uses a fixed seed (42) for reproducible results:
1. **First run**: Load from OSM and save files
2. **Fast run**: Load from saved files (much faster)
3. **Mixed run**: Load network from file, fetch fresh POIs
4. **Clean run**: Delete files and start fresh

## File Structure

The saved files will be stored in the `data/` directory:
```
data/
├── macau_network.pkl    # City street network
└── macau_pois.pkl       # Points of Interest
```

## Technical Details

- **Format**: Files are saved using Python's `pickle` module
- **Network**: Contains the complete NetworkX graph with nodes and edges
- **POIs**: Contains the dictionary structure with POIs organized by category
- **Compatibility**: Files are compatible across different runs and parameter combinations
- **Random Seed**: Affects all random number generators (Python, NumPy, Mesa)

## Performance Comparison

| Operation | First Run (OSM) | Subsequent Run (Files) | Speed Improvement |
|-----------|----------------|----------------------|-------------------|
| Load Network | ~30-60 seconds | ~1-2 seconds | **15-30x faster** |
| Load POIs | ~20-40 seconds | ~0.5-1 seconds | **20-40x faster** |
| **Total** | **~50-100 seconds** | **~1.5-3 seconds** | **~20-30x faster** |

## Best Practices

1. **Save on First Run**: Always use `--save-network` and `--save-pois` on your first run
2. **Load for Testing**: Use `--load-network` and `--load-pois` for rapid testing iterations
3. **Update When Needed**: Fetch fresh data occasionally to get updated POIs
4. **Backup Files**: Keep backup copies of your saved files
5. **Version Control**: Consider adding `data/*.pkl` to `.gitignore` due to file size
6. **Reproducibility**: Use `--seed` with the same value across test runs

## Troubleshooting

### File Not Found Error
```
Error: Network file not found: data/macau_network.pkl
```
**Solution**: Run with `--save-network` first to create the file.

### Loading Error
```
Error loading city network from file: ...
```
**Solution**: The file might be corrupted. Delete it and fetch fresh data from OSM.

### Different Results with Same Seed
If you get different results with the same seed:
1. Check if all files are loaded from the same source
2. Ensure no other random processes are affecting the simulation
3. Try clearing any cached data and rerunning with a fresh seed

### Large File Sizes
- Network files: ~5-15 MB
- POI files: ~1-5 MB
- Total: Usually under 20 MB

## Integration with Existing Features

The save/load functionality works seamlessly with all existing features:
- Parish filtering (`--parishes`)
- POI selection (`--essential-only`, `--all-pois`)
- Different movement behaviors (`--movement-behavior`)
- All other simulation parameters

Example with parishes and fixed seed:
```bash
# Save full data
python main.py --save-network data/macau_network.pkl --save-pois data/macau_pois.pkl --seed 42

# Load and filter to specific parishes (same seed for reproducibility)
python main.py --load-network data/macau_network.pkl --load-pois data/macau_pois.pkl --parishes "Sé" "Nossa Senhora de Fátima" --seed 42
``` 