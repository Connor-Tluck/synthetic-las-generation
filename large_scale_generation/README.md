# Large-Scale Synthetic Scene Generation

This folder contains the professional large-scale synthetic scene generation system, separate from the main synthetic scene generation toolkit.

## Overview

The large-scale generation system provides:
- **Realistic Scene Composition**: Urban planning principles for feature placement
- **Configurable Scenes**: Predefined and custom scene configurations
- **Feature Selection**: Choose how many of each infrastructure element to include
- **Batch Generation**: Generate multiple scene variations efficiently
- **Professional Output**: Comprehensive metadata and statistics

## Quick Start

```bash
# Generate a residential street scene
python generate_large_scale_scenes.py --config residential_street --preview

# Generate 5 commercial intersection scenes
python generate_large_scale_scenes.py --config commercial_intersection --count 5

# Use custom configuration
python generate_large_scale_scenes.py --custom-config configs/my_scene.json
```

## Files

- **`generate_large_scale_scenes.py`**: Main command-line interface
- **`synthetic_scenes.py`**: Modular scene generation components
- **`scene_config.py`**: Configuration management system
- **`scene_composer.py`**: Realistic scene composition engine
- **`configs/`**: Configuration files and examples
- **`docs/`**: Detailed documentation

## Predefined Configurations

- **`residential_street`**: Typical residential street with sidewalks, trees, utilities
- **`commercial_intersection`**: Busy commercial intersection with crosswalks, traffic control
- **`highway_section`**: Multi-lane highway with barriers, signage, utilities
- **`urban_park`**: City park with paths, benches, trees, recreational facilities

## Documentation

See `docs/LARGE_SCALE_GENERATION.md` for comprehensive documentation including:
- Detailed usage instructions
- Custom configuration examples
- Feature placement rules
- Troubleshooting guide

## Dependencies

- **numpy** (≥1.21.0): Numerical computations
- **laspy[lazrs]** (≥2.0.0): LAS/LAZ file I/O with compression
- **open3d** (≥0.15.0): 3D visualization (optional)

## Output

Generated scenes are saved to `large_scale_output/` with:
- LAZ/LAS point cloud files
- Comprehensive metadata JSON files
- Scene statistics and configuration tracking
