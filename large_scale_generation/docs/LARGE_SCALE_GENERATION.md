# Large-Scale Synthetic Scene Generation

This document describes the professional large-scale synthetic scene generation system that allows you to create realistic urban environments with configurable feature selection and realistic composition.

## Overview

The large-scale generation system consists of several modular components:

- **SceneGenerator**: Creates individual infrastructure elements
- **SceneConfigurationManager**: Manages predefined and custom scene configurations
- **SceneComposer**: Composes elements into realistic urban environments
- **LargeScaleGenerator**: Orchestrates the complete generation process

## Quick Start

### Basic Usage

```bash
# Generate a residential street scene
python generate_large_scale_scenes.py --config residential_street

# Generate 5 commercial intersection scenes
python generate_large_scale_scenes.py --config commercial_intersection --count 5

# List available configurations
python generate_large_scale_scenes.py --list-configs
```

### Custom Configuration

```bash
# Use a custom configuration file
python generate_large_scale_scenes.py --custom-config configs/my_scene.json --count 3

# Generate with custom output directory
python generate_large_scale_scenes.py --config urban_park --output-dir ./my_scenes
```

## Predefined Configurations

### Residential Street
- **Description**: Typical residential street with sidewalks, trees, and utilities
- **Size**: 50m × 100m
- **Features**: Street, sidewalks, curbs, trees, streetlights, utilities, fire hydrants, mailboxes
- **Complexity**: Standard

### Commercial Intersection
- **Description**: Busy commercial intersection with crosswalks, traffic control, and utilities
- **Size**: 80m × 80m
- **Features**: Multi-directional roads, crosswalks, sidewalks, traffic signs, streetlights, power poles, utilities, manholes, bollards
- **Complexity**: Detailed

### Highway Section
- **Description**: Multi-lane highway with barriers, signage, and utilities
- **Size**: 60m × 200m
- **Features**: Multi-lane roads, jersey barriers, guardrails, streetlights, utilities
- **Complexity**: Comprehensive

### Urban Park
- **Description**: City park with paths, benches, trees, and recreational facilities
- **Size**: 100m × 100m
- **Features**: Sidewalks, trees, benches, picnic tables, trash cans, bike racks, streetlights, fire hydrants
- **Complexity**: Detailed

## Custom Configuration

You can create custom scene configurations using JSON files. Here's the structure:

```json
{
  "name": "Custom Scene Name",
  "description": "Description of your custom scene",
  "complexity": "standard",
  "scene_size": {
    "width": 50.0,
    "length": 100.0,
    "height": 15.0
  },
  "features": {
    "feature_name": {
      "count": 5,
      "parameters": {
        "width": 8.0,
        "length": 10.0,
        "base_z": 0.0
      },
      "placement_rules": {
        "offset_from_road": 5.0,
        "side": "both"
      },
      "enabled": true
    }
  },
  "composition_rules": {
    "road_center": true,
    "sidewalk_symmetry": true
  },
  "output_settings": {
    "format": "laz",
    "compression": true,
    "include_metadata": true
  }
}
```

### Feature Configuration

Each feature in the configuration has:

- **count**: Number of instances to generate
- **parameters**: Parameters passed to the feature generator
- **placement_rules**: Rules for realistic placement
- **enabled**: Whether to include this feature

### Available Features

| Feature | Description | Key Parameters |
|---------|-------------|----------------|
| `street_patch` | Basic road surface | width, length, slope |
| `crosswalk` | Road with crosswalk markings | width, length |
| `curb_with_road` | Road with concrete curb | run_len, curb_w, curb_h |
| `sidewalk` | Concrete pedestrian walkway | width, length |
| `tree` | Vegetation with trunk and crown | (uses defaults) |
| `streetlight` | Road lighting fixture | (uses defaults) |
| `utility_cabinet` | Metal utility equipment box | (uses defaults) |
| `fire_hydrant` | Emergency water access | (uses defaults) |
| `mailbox_cluster` | Postal collection point | (uses defaults) |
| `power_pole_wires` | Utility pole with power lines | (uses defaults) |
| `stop_sign` | Traffic control sign | (uses defaults) |
| `manholes` | Utility access covers | count |
| `bollards` | Traffic protection posts | count |
| `jersey_barrier` | Concrete traffic barrier | run_len |
| `guardrail` | Metal safety barrier | run_len |
| `bench` | Park bench | (uses defaults) |
| `trash_can` | Waste receptacle | (uses defaults) |
| `bike_rack_u` | Bicycle parking | (uses defaults) |
| `picnic_table` | Outdoor seating | (uses defaults) |

### Placement Rules

Placement rules control how features are positioned in the scene:

- **offset_from_road**: Distance from road center
- **offset_from_sidewalk**: Distance from sidewalk
- **offset_from_curb**: Distance from curb
- **side**: "left", "right", "both", "alternating", "random"
- **spacing**: Distance between instances
- **intersection_corners**: Place at intersection corners
- **road_surface**: Place on road surface
- **shoulder**: Place on road shoulder
- **center_median**: Place in center median
- **near_entrances**: Place near scene entrances
- **near_benches**: Place near benches
- **grouped**: Group instances together
- **natural_clustering**: Use natural clustering for vegetation

## Command Line Options

### Basic Options
- `--config, -c`: Predefined configuration name
- `--custom-config`: Path to custom configuration JSON file
- `--list-configs`: List available predefined configurations
- `--count, -n`: Number of scenes to generate (default: 1)
- `--output-dir, -o`: Output directory (default: large_scale_output)

### Advanced Options
- `--format`: Output format (las/laz, default: laz)
- `--no-rgb`: Generate intensity-only data (no RGB colors)
- `--verbose, -v`: Verbose output

## Output Files

Each generated scene produces:

1. **Scene file**: `{scene_name}_{timestamp}.laz` - The point cloud data
2. **Metadata file**: `{scene_name}_{timestamp}_metadata.json` - Comprehensive metadata

### Metadata Contents

The metadata file includes:

- **Generation info**: Timestamp, version, configuration details
- **Scene info**: Point count, bounds, dimensions, density, format
- **Features**: Enabled features, counts, types
- **Classification summary**: Point classification statistics
- **Configuration**: Complete scene configuration used

## Realistic Composition

The system follows urban planning principles for realistic scene composition:

### Priority System
Features are placed in priority order:
1. Infrastructure (roads, sidewalks, curbs)
2. Major utilities (power poles, streetlights, cabinets)
3. Safety elements (barriers, bollards)
4. Traffic control (signs, markings)
5. Vegetation (trees)
6. Street furniture (benches, trash cans)
7. Utilities and services (hydrants, mailboxes)
8. Accessibility features (ramps, stairs)
9. Complex features (medians, furniture)

### Spatial Relationships
- Roads are centered in the scene
- Sidewalks are placed alongside roads with appropriate offsets
- Utilities are positioned with realistic spacing
- Vegetation follows natural clustering patterns
- Street furniture is placed near pedestrian areas

### Collision Avoidance
The system includes basic collision detection to prevent overlapping features.

## Performance Considerations

### Scene Size
- Larger scenes take longer to generate
- Point count scales with scene area and feature density
- Recommended maximum scene size: 200m × 200m

### Feature Count
- More features increase generation time
- Complex features (trees, power poles) are more expensive
- Recommended maximum: 100 total feature instances

### Batch Generation
- Multiple scenes can be generated efficiently
- Each scene is independent and can be parallelized
- Metadata is generated for each scene

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Configuration Errors**: Validate JSON syntax in custom configs
3. **Memory Issues**: Reduce scene size or feature count
4. **File Permissions**: Ensure write access to output directory

### Debug Mode

Use `--verbose` flag for detailed error information:

```bash
python generate_large_scale_scenes.py --config residential_street --verbose
```

## Examples

### Example 1: Residential Street
```bash
python generate_large_scale_scenes.py --config residential_street --count 3
```

### Example 2: Custom Commercial Scene
```bash
python generate_large_scale_scenes.py --custom-config configs/commercial.json --count 5 --output-dir ./commercial_scenes
```

### Example 3: High-Density Urban Scene
```bash
python generate_large_scale_scenes.py --config commercial_intersection --count 10 --format laz --output-dir ./urban_data
```

## Integration

The large-scale generation system can be integrated into larger workflows:

```python
from generate_large_scale_scenes import LargeScaleGenerator
from scene_config import SceneConfigurationManager

# Initialize generator
config_manager = SceneConfigurationManager()
generator = LargeScaleGenerator()

# Get configuration
config = config_manager.get_predefined_config('residential_street')

# Generate scene
result = generator.generate_scene(config, 'my_scene')

if result['success']:
    print(f"Generated {result['output_file']}")
    print(f"Point count: {result['point_count']:,}")
```

## Best Practices

1. **Start Simple**: Begin with predefined configurations
2. **Test Small**: Generate single scenes before batch processing
3. **Validate Configs**: Check custom configurations with small scenes first
4. **Monitor Resources**: Watch memory usage for large scenes
5. **Use Metadata**: Leverage metadata files for scene analysis
6. **Version Control**: Keep configuration files in version control
