#!/usr/bin/env python3
"""
Density Test Grid Generator

Creates a systematic grid of synthetic point cloud scenes for testing different
point densities. Features 20 different urban infrastructure elements across
5 density columns (100, 300, 500, 700, 1000 points per square meter).

This tool is designed for:
- Algorithm testing across different point densities
- Performance benchmarking
- Quality assessment at various resolutions
- Machine learning training data generation
"""

import math
import json
import numpy as np
import laspy
from pathlib import Path
import time

try:
    import open3d as o3d
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False

# Import the existing scene generation functions
from generate_point_cloud_sandbox import (
    make_grid, jitter, clipped_intensity, color_arrays_for, stack_fields,
    translate, ensure_cls, attach_rgb, make_plane, make_box, make_cylinder,
    make_wire, make_stairs, make_curb, make_crosswalk, make_barrier_jersey,
    make_street_patch, make_manholes_and_grates, make_bench, make_tree,
    make_sidewalk, make_power_pole_and_lines, make_parking_bumpers,
    make_speed_hump, make_guardrail, make_utility_box, make_phone_cabinet,
    make_bollards, make_driveway_crown, make_stop_sign, make_streetlight,
    make_fire_hydrant, make_mailbox_cluster, make_bike_rack_u, make_trash_can,
    make_picnic_table, make_median_island_with_curbs, make_ped_ramp_with_tactile,
    make_storm_inlet_grate, MAT_INT, MAT_RGB_255, CLASS, RNG, XY_JITTER, Z_JITTER
)

# -----------------------------
# Configuration
# -----------------------------
OUT_DIR = Path("density_test_grid_output")
WRITE_LAZ = True
SHOW_PREVIEW = True
ADD_RGB = True

# Density test configuration
DENSITY_LEVELS = [50, 100, 400, 1000, 5000]  # points per square meter
FEATURE_SIZE = 8.0  # meters (square area for each feature)
GRID_SPACING = 20.0  # meters between features (increased spacing)
LABEL_HEIGHT = 2.0  # height above features for density labels
NUMERICAL_LABEL_HEIGHT = 1.0  # height for numerical density labels at bottom

# -----------------------------
# Density-specific functions
# -----------------------------

def get_spacing_for_density(density):
    """Calculate grid spacing for target density."""
    return (1.0 / density) ** 0.5

def make_density_label(density, x, y, z, spacing):
    """Create a text label for density level."""
    # Create simple geometric label (rectangular block with density number)
    label_width = 1.5
    label_height = 0.3
    label_depth = 0.1
    
    # Create the label base
    label = make_box(label_width, label_height, label_depth, z, "paint_white", "unclassified", spacing)
    label = translate(label, x - label_width/2, y - label_height/2, 0)
    
    # Add density number as simple geometric representation
    # For simplicity, we'll use the density value as intensity
    label["intensity"] = np.full(label["x"].size, min(density * 10, 65535), dtype=np.uint16)
    
    return label

def make_numerical_density_label(density, x, y, z, spacing):
    """Create human-readable numerical density label using point cloud."""
    # Convert density number to string
    density_str = str(density)
    
    # Create points to form the numbers
    all_points = []
    all_colors = []
    all_intensities = []
    all_classes = []
    
    # Character dimensions (increased for better visibility)
    char_width = 1.5
    char_height = 2.0
    char_spacing = 0.3
    
    # Starting position for the number
    start_x = x - (len(density_str) * (char_width + char_spacing)) / 2
    
    for i, digit in enumerate(density_str):
        char_x = start_x + i * (char_width + char_spacing)
        digit_points = create_digit_points(digit, char_x, y, z, char_width, char_height, spacing)
        
        if digit_points is not None:
            all_points.append(digit_points)
            # Use bright white color for visibility
            all_colors.append(np.full((len(digit_points), 3), [255, 255, 255], dtype=np.uint16))
            all_intensities.append(np.full(len(digit_points), 65535, dtype=np.uint16))
            all_classes.append(np.full(len(digit_points), CLASS["unclassified"], dtype=np.uint8))
    
    if all_points:
        # Combine all digit points
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        combined_intensities = np.hstack(all_intensities)
        combined_classes = np.hstack(all_classes)
        
        return {
            "x": combined_points[:, 0],
            "y": combined_points[:, 1], 
            "z": combined_points[:, 2],
            "red": combined_colors[:, 0],
            "green": combined_colors[:, 1],
            "blue": combined_colors[:, 2],
            "intensity": combined_intensities,
            "class": combined_classes
        }
    else:
        # Fallback to simple box if digit creation fails
        return make_density_label(density, x, y, z, spacing)

def create_digit_points(digit, x, y, z, width, height, spacing):
    """Create point cloud representation of a single digit using 7-segment display."""
    # 7-segment display layout:
    #   aaa
    #  f   b
    #  f   b
    #   ggg
    #  e   c
    #  e   c
    #   ddd
    
    # Define segment endpoints (normalized coordinates 0-1)
    segments = {
        'a': [(0.1, 0.8), (0.9, 0.8)],  # top
        'b': [(0.9, 0.8), (0.9, 0.5)],  # top-right
        'c': [(0.9, 0.5), (0.9, 0.2)],  # bottom-right
        'd': [(0.1, 0.2), (0.9, 0.2)],  # bottom
        'e': [(0.1, 0.5), (0.1, 0.2)],  # bottom-left
        'f': [(0.1, 0.8), (0.1, 0.5)],  # top-left
        'g': [(0.1, 0.5), (0.9, 0.5)]   # middle
    }
    
    # Define which segments are lit for each digit
    digit_segments = {
        '0': ['a', 'b', 'c', 'd', 'e', 'f'],
        '1': ['b', 'c'],
        '2': ['a', 'b', 'g', 'e', 'd'],
        '3': ['a', 'b', 'g', 'c', 'd'],
        '4': ['f', 'g', 'b', 'c'],
        '5': ['a', 'f', 'g', 'c', 'd'],
        '6': ['a', 'f', 'g', 'e', 'd', 'c'],
        '7': ['a', 'b', 'c'],
        '8': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
        '9': ['a', 'b', 'c', 'd', 'f', 'g']
    }
    
    if digit not in digit_segments:
        return None
    
    points = []
    segment_thickness = 0.15  # Thickness of each segment
    
    for segment_name in digit_segments[digit]:
        if segment_name in segments:
            start, end = segments[segment_name]
            
            # Create a thick line for the segment
            # Calculate perpendicular direction for thickness
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Normalize direction
                dx /= length
                dy /= length
                
                # Perpendicular direction for thickness
                perp_x = -dy
                perp_y = dx
                
                # Create points along the segment with thickness
                num_points_along = max(3, int(length * width / spacing))
                num_points_thick = max(2, int(segment_thickness * height / spacing))
                
                for i in range(num_points_along):
                    t = i / (num_points_along - 1) if num_points_along > 1 else 0
                    center_x = start[0] + t * (end[0] - start[0])
                    center_y = start[1] + t * (end[1] - start[1])
                    
                    # Add thickness
                    for j in range(num_points_thick):
                        thick_t = (j / (num_points_thick - 1) - 0.5) if num_points_thick > 1 else 0
                        px = x + (center_x + thick_t * perp_x * segment_thickness) * width
                        py = y + (center_y + thick_t * perp_y * segment_thickness) * height
                        pz = z
                        points.append([px, py, pz])
    
    return np.array(points) if points else None

def make_feature_with_density(feature_func, density, **kwargs):
    """Generate a feature with specific density."""
    # Calculate spacing for target density
    spacing = get_spacing_for_density(density)
    
    # Generate the feature with custom spacing
    if 'spacing' in kwargs:
        # If spacing is already specified, use the minimum for density control
        kwargs['spacing'] = min(kwargs['spacing'], spacing)
    else:
        kwargs['spacing'] = spacing
    
    return feature_func(**kwargs)

# -----------------------------
# Feature library for testing
# -----------------------------

def build_density_test_features():
    """Build the library of features for density testing."""
    features = []
    
    # Road infrastructure
    features.append(("street_patch", lambda **kwargs: make_street_patch(
        kwargs.get('width', 8.0), kwargs.get('length', 8.0), 
        kwargs.get('base_z', 0.0), kwargs.get('slope', (0.0, 0.01))
    )))
    
    features.append(("crosswalk", lambda **kwargs: make_crosswalk(
        kwargs.get('width', 6.0), kwargs.get('length', 8.0), 
        kwargs.get('base_z', 0.0)
    )))
    
    features.append(("curb_with_road", lambda **kwargs: make_curb(
        kwargs.get('run_len', 8.0), kwargs.get('curb_w', 0.3), 
        kwargs.get('curb_h', 0.15), kwargs.get('base_z', 0.0), 
        kwargs.get('road_slope', (0.0, 0.002))
    )))
    
    features.append(("sidewalk", lambda **kwargs: make_sidewalk(
        kwargs.get('width', 3.0), kwargs.get('length', 8.0), 
        kwargs.get('base_z', 0.0)
    )))
    
    # Barriers and safety
    features.append(("jersey_barrier", lambda **kwargs: make_barrier_jersey(
        kwargs.get('run_len', 8.0), kwargs.get('base_z', 0.0)
    )))
    
    features.append(("guardrail", lambda **kwargs: make_guardrail(
        kwargs.get('run_len', 8.0), kwargs.get('base_z', 0.0)
    )))
    
    features.append(("bollards", lambda **kwargs: make_bollards(
        kwargs.get('count', 4), kwargs.get('base_z', 0.0)
    )))
    
    # Utilities and infrastructure
    features.append(("power_pole_wires", lambda **kwargs: make_power_pole_and_lines(
        kwargs.get('base_z', 0.0)
    )))
    
    features.append(("streetlight", lambda **kwargs: make_streetlight(
        kwargs.get('base_z', 0.0)
    )))
    
    features.append(("utility_cabinet", lambda **kwargs: make_utility_box(
        kwargs.get('base_z', 0.0)
    )))
    
    features.append(("phone_cabinet", lambda **kwargs: make_phone_cabinet(
        kwargs.get('base_z', 0.0)
    )))
    
    features.append(("manholes", lambda **kwargs: make_manholes_and_grates(
        kwargs.get('count', 2), kwargs.get('base_z', 0.0)
    )))
    
    features.append(("storm_inlet_grate", lambda **kwargs: make_storm_inlet_grate(
        kwargs.get('base_z', 0.0)
    )))
    
    # Traffic control
    features.append(("stop_sign", lambda **kwargs: make_stop_sign(
        kwargs.get('base_z', 0.0)
    )))
    
    features.append(("parking_bumpers", lambda **kwargs: make_parking_bumpers(
        kwargs.get('count', 3), kwargs.get('base_z', 0.0)
    )))
    
    features.append(("speed_hump", lambda **kwargs: make_speed_hump(
        kwargs.get('width', 6.0), kwargs.get('length', 8.0), 
        kwargs.get('height', 0.12), kwargs.get('base_z', 0.0)
    )))
    
    # Vegetation and landscaping
    features.append(("tree", lambda **kwargs: make_tree(
        kwargs.get('base_z', 0.0)
    )))
    
    # Street furniture
    features.append(("bench", lambda **kwargs: make_bench(
        kwargs.get('base_z', 0.0)
    )))
    
    features.append(("trash_can", lambda **kwargs: make_trash_can(
        kwargs.get('base_z', 0.0)
    )))
    
    features.append(("bike_rack_u", lambda **kwargs: make_bike_rack_u(
        kwargs.get('base_z', 0.0)
    )))
    
    features.append(("picnic_table", lambda **kwargs: make_picnic_table(
        kwargs.get('base_z', 0.0)
    )))
    
    return features

# -----------------------------
# Grid generation
# -----------------------------

def generate_density_test_grid():
    """Generate the complete density test grid."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    features = build_density_test_features()
    total_rows = len(features) + 2 + 1  # features + 2 stair rows + 1 label row
    print(f"Generating density test grid with {len(features)} features and {len(DENSITY_LEVELS)} density levels")
    print(f"Grid size: {total_rows} rows × {len(DENSITY_LEVELS)} columns")
    print(f"  - {len(features)} feature rows")
    print(f"  - 2 stair structure rows") 
    print(f"  - 1 numerical label row")
    
    all_scenes = []
    grid_metadata = []
    
    # Generate each feature at each density
    for row, (feature_name, feature_func) in enumerate(features):
        print(f"Processing row {row+1}/{len(features)}: {feature_name}")
        
        for col, density in enumerate(DENSITY_LEVELS):
            # Calculate position in grid
            x = col * GRID_SPACING
            y = row * GRID_SPACING
            
            # Generate feature with specific density
            try:
                scene_data = make_feature_with_density(feature_func, density)
                scene_data = ensure_cls(scene_data)
                
                # Translate to grid position
                scene_data = translate(scene_data, x, y, 0.0)
                
                # Add density label
                label = make_density_label(density, x, y, LABEL_HEIGHT, get_spacing_for_density(density))
                label = translate(label, 0, 0, 0)  # Already positioned correctly
                
                # Combine feature and label
                combined_scene = stack_fields([scene_data, label])
                
                all_scenes.append(combined_scene)
                
                # Store metadata
                grid_metadata.append({
                    "row": row,
                    "col": col,
                    "feature_name": feature_name,
                    "density": density,
                    "position": {"x": x, "y": y, "z": 0.0},
                    "point_count": len(combined_scene["x"])
                })
                
            except Exception as e:
                print(f"  Warning: Failed to generate {feature_name} at density {density}: {e}")
                continue
    
    # Add 2 stair structures to each column at the bottom
    print("Adding stair structures to each column...")
    stairs_row_start = len(features)  # Start after all features
    
    for col, density in enumerate(DENSITY_LEVELS):
        x = col * GRID_SPACING
        
        # Add 2 stair structures per column
        for stair_idx in range(2):
            y = (stairs_row_start + stair_idx) * GRID_SPACING
            
            try:
                # Generate stairs with specific density
                stairs_data = make_feature_with_density(make_stairs, density, 
                                                      width=3.0, length=4.0, 
                                                      step_count=5, step_height=0.15, 
                                                      base_z=0.0)
                stairs_data = ensure_cls(stairs_data)
                
                # Translate to grid position
                stairs_data = translate(stairs_data, x, y, 0.0)
                
                # Add density label
                label = make_density_label(density, x, y, LABEL_HEIGHT, get_spacing_for_density(density))
                label = translate(label, 0, 0, 0)
                
                # Combine stairs and label
                combined_stairs = stack_fields([stairs_data, label])
                
                all_scenes.append(combined_stairs)
                
                # Store metadata
                grid_metadata.append({
                    "row": stairs_row_start + stair_idx,
                    "col": col,
                    "feature_name": "stairs",
                    "density": density,
                    "position": {"x": x, "y": y, "z": 0.0},
                    "point_count": len(combined_stairs["x"])
                })
                
            except Exception as e:
                print(f"  Warning: Failed to generate stairs at density {density}: {e}")
                continue
    
    # Add numerical density labels at the bottom of each column
    print("Adding numerical density labels...")
    label_row = stairs_row_start + 2  # After the 2 stair rows
    
    for col, density in enumerate(DENSITY_LEVELS):
        x = col * GRID_SPACING
        y = label_row * GRID_SPACING
        
        try:
            # Create numerical density label
            numerical_label = make_numerical_density_label(density, x, y, NUMERICAL_LABEL_HEIGHT, 
                                                         get_spacing_for_density(density))
            
            if numerical_label is not None:
                all_scenes.append(numerical_label)
                
                # Store metadata
                grid_metadata.append({
                    "row": label_row,
                    "col": col,
                    "feature_name": "density_label",
                    "density": density,
                    "position": {"x": x, "y": y, "z": NUMERICAL_LABEL_HEIGHT},
                    "point_count": len(numerical_label["x"])
                })
                
        except Exception as e:
            print(f"  Warning: Failed to generate numerical label for density {density}: {e}")
            continue
    
    # Combine all scenes
    print("Combining all scenes...")
    combined_scene = stack_fields(all_scenes)
    
    # Write output files
    timestamp = int(time.time())
    
    # Write LAS/LAZ file
    las_file = OUT_DIR / f"density_test_grid_{timestamp}.las"
    write_las(las_file, combined_scene)
    
    if WRITE_LAZ:
        try:
            laz_file = OUT_DIR / f"density_test_grid_{timestamp}.laz"
            las = laspy.read(las_file)
            las.write(laz_file)
            las_file.unlink()
            final_file = laz_file
        except Exception as e:
            print(f"Warning: Could not compress to LAZ ({e}), keeping LAS file")
            final_file = las_file
    else:
        final_file = las_file
    
    # Write metadata
    metadata = {
        "generation_info": {
            "timestamp": timestamp,
            "generator": "density_test_grid",
            "version": "1.0.0"
        },
        "grid_config": {
            "features": len(features),
            "density_levels": DENSITY_LEVELS,
            "feature_size": FEATURE_SIZE,
            "grid_spacing": GRID_SPACING,
            "label_height": LABEL_HEIGHT
        },
        "scene_info": {
            "total_points": len(combined_scene["x"]),
            "output_file": final_file.name,
            "has_rgb": ADD_RGB,
            "format": "LAZ" if WRITE_LAZ else "LAS"
        },
        "grid_data": grid_metadata
    }
    
    metadata_file = OUT_DIR / f"density_test_grid_{timestamp}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Write grid legend
    legend_data = {
        "density_levels": DENSITY_LEVELS,
        "features": [{"index": i, "name": name} for i, (name, _) in enumerate(features)],
        "grid_info": {
            "rows": len(features),
            "columns": len(DENSITY_LEVELS),
            "spacing": GRID_SPACING
        }
    }
    
    legend_file = OUT_DIR / f"density_test_grid_{timestamp}_legend.json"
    with open(legend_file, 'w') as f:
        json.dump(legend_data, f, indent=2)
    
    print(f"✓ Generated {final_file.name}")
    print(f"✓ Points: {len(combined_scene['x']):,}")
    print(f"✓ Metadata: {metadata_file.name}")
    print(f"✓ Legend: {legend_file.name}")
    
    # Calculate and display statistics
    unique_classes, class_counts = np.unique(combined_scene["cls"], return_counts=True)
    print(f"\nClassification summary:")
    for c, n in zip(unique_classes.tolist(), class_counts.tolist()):
        print(f"  Class {c}: {n:,} points")
    
    # Show preview if requested
    if SHOW_PREVIEW and HAVE_O3D:
        print("\nOpening 3D preview...")
        points = np.vstack([combined_scene["x"], combined_scene["y"], combined_scene["z"]]).T
        
        if ADD_RGB and all(k in combined_scene for k in ["red", "green", "blue"]):
            colors = np.vstack([
                combined_scene["red"],
                combined_scene["green"],
                combined_scene["blue"]
            ]).T.astype(np.float32) / 65535.0
        else:
            colors = (combined_scene["intensity"].astype(np.float32) / 65535.0).reshape(-1, 1)
            colors = np.repeat(colors, 3, axis=1)
        
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name="Density Test Grid")
    
    return final_file, metadata_file, legend_file

# -----------------------------
# LAS writing function
# -----------------------------

def write_las(filepath, pts):
    """Write point cloud data to LAS/LAZ file."""
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)

    las.x = pts["x"]
    las.y = pts["y"]
    las.z = pts["z"]
    las.intensity = pts["intensity"]

    cls = pts.get("cls", pts.get("class"))
    if cls is None:
        raise KeyError("points missing classification key cls or class")
    las.classification = cls.astype(np.uint8)

    if ADD_RGB and all(k in pts for k in ["red", "green", "blue"]):
        las.red = pts["red"].astype(np.uint16)
        las.green = pts["green"].astype(np.uint16)
        las.blue = pts["blue"].astype(np.uint16)

    las.return_number = np.ones(pts["x"].size, dtype=np.uint8)
    las.number_of_returns = np.ones(pts["x"].size, dtype=np.uint8)
    las.write(filepath)

# -----------------------------
# Main
# -----------------------------

def main():
    """Main function."""
    print("Density Test Grid Generator")
    print("=" * 50)
    print(f"Features: 20 urban infrastructure elements")
    print(f"Densities: {DENSITY_LEVELS} points/m²")
    print(f"Grid: 20 rows × 5 columns")
    print(f"Output: {OUT_DIR}")
    print()
    
    start_time = time.time()
    
    try:
        final_file, metadata_file, legend_file = generate_density_test_grid()
        
        generation_time = time.time() - start_time
        print(f"\nGeneration complete in {generation_time:.1f}s")
        print(f"Output directory: {OUT_DIR}")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
