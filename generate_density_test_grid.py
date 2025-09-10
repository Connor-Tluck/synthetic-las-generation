#!/usr/bin/env python3
"""
Clean Density Test Grid Generator

Creates a systematic grid of clean synthetic point cloud scenes using only
well-working geometries from the sandbox. Features clean, single objects
across 5 density columns (50, 100, 400, 1000, 5000 points per square meter).

This tool focuses on clean, individual geometries without complex multi-part objects.
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
    make_plane, make_box, make_cylinder, make_stairs, make_tree, make_bench,
    make_utility_box, make_phone_cabinet, make_stop_sign, make_streetlight,
    make_fire_hydrant, make_mailbox_cluster, make_bike_rack_u, make_trash_can,
    make_picnic_table, make_storm_inlet_grate, make_speed_hump,
    translate, stack_fields, ensure_cls, MAT_INT, MAT_RGB_255, CLASS, RNG, XY_JITTER, Z_JITTER
)

# -----------------------------
# Configuration
# -----------------------------
OUT_DIR = Path("density_test_grid_output")
WRITE_LAZ = True
SHOW_PREVIEW = True
ADD_RGB = True

# Density test configuration
DENSITY_LEVELS = [10, 50, 200, 1000, 5000]  # points per square meter - wider range for more obvious differences
FEATURE_SIZE = 8.0  # meters (square area for each feature)
GRID_SPACING = 20.0  # meters between features
LABEL_HEIGHT = 2.0  # height above features for density labels
NUMERICAL_LABEL_HEIGHT = 1.0  # height for numerical density labels at bottom

# -----------------------------
# Density-specific functions
# -----------------------------

def get_spacing_for_density(density):
    """Calculate grid spacing for target density."""
    return (1.0 / density) ** 0.5

def make_density_label(density, x, y, z, spacing):
    """Create a simple text label for density level."""
    # Return None to skip density labels - they're causing the weird repeating bars
    return None

def create_digit_points(digit, x, y, z, width, height, spacing):
    """Create point cloud representation of a single digit using thick 7-segment display."""
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
    segment_thickness = 0.3  # Thickness of each segment (30% of character size)

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
                num_points_thick = max(3, int(segment_thickness * height / spacing))

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

def make_numerical_density_label(density, x, y, z, spacing):
    """Create human-readable numerical density label using point cloud."""
    # Convert density number to string
    density_str = str(density)
    
    # Create points to form the numbers
    all_points = []
    all_colors = []
    all_intensities = []
    all_classes = []
    
    # Character dimensions (appropriately sized for visibility)
    char_width = 2.5
    char_height = 3.5
    char_spacing = 0.6
    
    # Starting position for the number
    start_x = x - (len(density_str) * (char_width + char_spacing)) / 2
    
    for i, digit in enumerate(density_str):
        char_x = start_x + i * (char_width + char_spacing)
        digit_points = create_digit_points(digit, char_x, y, z, char_width, char_height, spacing)
        
        if digit_points is not None:
            all_points.append(digit_points)
            # Use white color for visibility
            colors = np.tile([1.0, 1.0, 1.0], (len(digit_points), 1))
            all_colors.append(colors)
            # High intensity for visibility
            intensities = np.full(len(digit_points), 65535, dtype=np.uint16)
            all_intensities.append(intensities)
            # Classification
            classes = np.full(len(digit_points), CLASS["unclassified"], dtype=np.uint8)
            all_classes.append(classes)
    
    if not all_points:
        return None
    
    # Combine all digit points
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    combined_intensities = np.hstack(all_intensities)
    combined_classes = np.hstack(all_classes)
    
    # Create the data structure
    result = {
        "x": combined_points[:, 0],
        "y": combined_points[:, 1], 
        "z": combined_points[:, 2],
        "intensity": combined_intensities,
        "class": combined_classes
    }
    
    # Add RGB colors
    if ADD_RGB:
        result["red"] = (combined_colors[:, 0] * 65535).astype(np.uint16)
        result["green"] = (combined_colors[:, 1] * 65535).astype(np.uint16)
        result["blue"] = (combined_colors[:, 2] * 65535).astype(np.uint16)
    
    return result

def make_feature_with_density(feature_func, density, **kwargs):
    """Generate a feature with specific density."""
    # Calculate spacing for target density
    spacing = get_spacing_for_density(density)
    
    # Always pass the spacing parameter to the feature function
    kwargs['spacing'] = spacing
    
    # Debug output
    print(f"    Density: {density} pts/m², Spacing: {spacing:.4f}m")
    
    # Generate the feature with density-controlled spacing
    result = feature_func(**kwargs)
    
    # Debug: show actual point count and spacing calculation
    if result and 'x' in result:
        point_count = len(result['x'])
        # Calculate expected points for 8x8 area
        expected_points = int(8.0 / spacing) * int(8.0 / spacing)
        print(f"      -> Generated {point_count} points (expected ~{expected_points})")
    
    return result

# -----------------------------
# Clean feature definitions (using only well-working geometries)
# -----------------------------

def build_clean_features():
    """Build list of clean, single-object features from sandbox."""
    features = []
    
    # Basic geometric shapes (these work perfectly) - using consistent 8x8 size for fair comparison
    features.append(("plane_8x8", lambda **kwargs: make_plane(
        kwargs.get('width', 8.0), kwargs.get('length', 8.0), 
        kwargs.get('base_z', 0.0), "asphalt", "road_surface", kwargs.get('spacing', 0.05)
    )))
    
    features.append(("box_2x2x1", lambda **kwargs: make_box(
        kwargs.get('length', 2.0), kwargs.get('width', 2.0), kwargs.get('height', 1.0),
        kwargs.get('base_z', 0.0), "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    features.append(("cylinder_1m_high", lambda **kwargs: make_cylinder(
        kwargs.get('radius', 0.5), kwargs.get('height', 1.0), 
        kwargs.get('base_z', 0.0), "metal", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    # Just a few more basic shapes for testing
    features.append(("stairs", lambda **kwargs: make_stairs(
        kwargs.get('width', 2.0), kwargs.get('depth', 0.3), kwargs.get('step_h', 0.15),
        kwargs.get('n_steps', 5), kwargs.get('base_z', 0.0), "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    features.append(("tree", lambda **kwargs: make_tree(
        kwargs.get('base_z', 0.0), kwargs.get('spacing', 0.05)
    )))
    
    return features

# -----------------------------
# Main generation function
# -----------------------------

def generate_clean_density_test_grid():
    """Generate the clean density test grid."""
    print("Generating Clean Density Test Grid")
    print("=" * 50)
    
    # Create output directory
    OUT_DIR.mkdir(exist_ok=True)
    
    # Build clean features
    features = build_clean_features()
    print(f"Using {len(features)} clean features across {len(DENSITY_LEVELS)} density levels")
    
    # Generate grid
    all_scenes = []
    grid_metadata = []
    
    print(f"Generating clean density test grid with {len(features)} features and {len(DENSITY_LEVELS)} density levels")
    
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
                
                # Add density label (skip if None)
                label = make_density_label(density, x, y, LABEL_HEIGHT, get_spacing_for_density(density))
                
                # Combine feature and label (if label exists)
                if label is not None:
                    combined_scene = stack_fields([scene_data, label])
                else:
                    combined_scene = scene_data
                
                all_scenes.append(combined_scene)
                
                # Store metadata
                grid_metadata.append({
                    "row": row,
                    "col": col,
                    "feature_name": feature_name,
                    "density": density,
                    "position": {"x": x, "y": y, "z": 0.0}
                })
                
            except Exception as e:
                print(f"  Error generating {feature_name} at density {density}: {e}")
                continue
    
    # Add numerical density labels at the bottom of each column
    print("Adding numerical density labels...")
    for col, density in enumerate(DENSITY_LEVELS):
        x = col * GRID_SPACING
        y = -GRID_SPACING  # Position below the grid
        z = NUMERICAL_LABEL_HEIGHT
        
        # Create numerical label
        label = make_numerical_density_label(density, x, y, z, get_spacing_for_density(density))
        if label is not None:
            all_scenes.append(label)
    
    # Combine all scenes
    print("Combining all scenes...")
    if not all_scenes:
        print("No scenes generated!")
        return
    
    combined_scene = stack_fields(all_scenes)
    
    # Save metadata
    metadata_file = OUT_DIR / "enhanced_grid_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "density_levels": DENSITY_LEVELS,
            "grid_spacing": GRID_SPACING,
            "features": [name for name, _ in features],
            "grid_metadata": grid_metadata,
            "total_points": len(combined_scene["x"]),
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    # Save point cloud
    print("Saving point cloud...")
    output_file = OUT_DIR / "enhanced_density_test_grid.laz"
    
    try:
        # Create LAS file
        las = laspy.create(point_format=3, file_version="1.4")
        las.x = combined_scene["x"]
        las.y = combined_scene["y"] 
        las.z = combined_scene["z"]
        las.intensity = combined_scene["intensity"]
        las.classification = combined_scene.get("cls", combined_scene.get("class"))
        
        if ADD_RGB and all(k in combined_scene for k in ["red", "green", "blue"]):
            las.red = combined_scene["red"]
            las.green = combined_scene["green"]
            las.blue = combined_scene["blue"]
        
        # Write file
        if WRITE_LAZ:
            las.write(output_file)
        else:
            las.write(str(output_file).replace('.laz', '.las'))
        
        print(f"Saved: {output_file}")
        print(f"Total points: {len(combined_scene['x']):,}")
        
    except Exception as e:
        print(f"Error saving LAZ file: {e}")
        # Fallback to LAS
        try:
            output_file = output_file.with_suffix('.las')
            las.write(output_file)
            print(f"Saved as LAS: {output_file}")
        except Exception as e2:
            print(f"Error saving LAS file: {e2}")
            return
    
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

        # Create visualizer with better zoom controls
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Clean Density Test Grid - Enhanced Zoom", width=1200, height=800)
        vis.add_geometry(pcd)

        # Configure render options for better visualization
        render_option = vis.get_render_option()
        render_option.point_size = 2.0  # Larger point size for better visibility
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background

        # Set up camera with better initial view
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(0.3)  # Start more zoomed out

        print("  Enhanced preview with better zoom controls:")
        print("  - Mouse wheel: Zoom in/out")
        print("  - Right-click + drag: Pan")
        print("  - Left-click + drag: Rotate")
        print("  - Close window to continue...")

        vis.run()
        vis.destroy_window()
    
    elif SHOW_PREVIEW and not HAVE_O3D:
        print("Open3D not available for preview")
    
    print("\nClean density test grid generation complete!")
    print(f"Output directory: {OUT_DIR}")

if __name__ == "__main__":
    generate_clean_density_test_grid()
