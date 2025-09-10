#!/usr/bin/env python3
"""
Geometric Shapes Test Grid Generator

Creates a focused test grid of geometric shapes specifically designed for drawing tool testing.
Features a variety of geometric primitives and complex shapes with high point density (5000 pts/m²).

This tool focuses on geometric shapes that challenge drawing, measurement, and pattern recognition tools.
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
    make_plane, make_box, make_cylinder, make_stairs,
    make_sphere, make_stepped_pyramid, make_hexagon, make_cross,
    translate, stack_fields, ensure_cls, MAT_INT, MAT_RGB_255, CLASS, RNG, XY_JITTER, Z_JITTER
)

# -----------------------------
# Configuration
# -----------------------------
OUT_DIR = Path("geometric_shapes_output")
WRITE_LAZ = True
SHOW_PREVIEW = True
ADD_RGB = True

# High density configuration for detailed geometric testing
DENSITY_LEVELS = [5000]  # Single high density level for all shapes
FEATURE_SIZE = 12.0  # meters (square area for each feature)
GRID_SPACING = 50.0  # meters between features (increased for larger planes)
LABEL_HEIGHT = 3.0  # height above features for labels
Z_OFFSET = 0.0  # Z offset for all elements

# -----------------------------
# Additional Geometric Shape Functions
# -----------------------------

def make_rotated_plane(width=8.0, length=8.0, base_z=0.0, material="concrete", classification="unclassified", point_spacing=0.05, rotation_angle=15):
    """Create a plane rotated around the Y axis."""
    # Create regular plane first
    plane = make_plane(width, length, base_z, material, classification, point_spacing)
    
    # Apply rotation around Y axis
    angle_rad = np.radians(rotation_angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotate points around Y axis
    x_rot = plane["x"] * cos_a - plane["z"] * sin_a
    z_rot = plane["x"] * sin_a + plane["z"] * cos_a
    
    plane["x"] = x_rot
    plane["z"] = z_rot
    
    return plane

def make_cone(radius=2.0, height=3.0, base_z=0.0, material="concrete", classification="unclassified", point_spacing=0.05):
    """Create a cone shape."""
    points = []
    intensities = []
    classes = []
    colors = []
    
    # Create cone using radial distribution
    n_radial = max(20, int(2 * np.pi * radius / point_spacing))
    n_height = max(10, int(height / point_spacing))
    
    for h in range(n_height):
        z = base_z + (h / (n_height - 1)) * height
        current_radius = radius * (1 - h / (n_height - 1))
        
        for r in range(n_radial):
            angle = (r / n_radial) * 2 * np.pi
            x = current_radius * np.cos(angle)
            y = current_radius * np.sin(angle)
            
            points.append([x, y, z])
            intensities.append(MAT_INT[material][0])
            classes.append(CLASS[classification])
            colors.append(MAT_RGB_255[material])
    
    return {
        "x": np.array([p[0] for p in points]),
        "y": np.array([p[1] for p in points]),
        "z": np.array([p[2] for p in points]),
        "intensity": np.array(intensities, dtype=np.uint16),
        "class": np.array(classes, dtype=np.uint8),
        "red": np.array([c[0] for c in colors], dtype=np.uint16),
        "green": np.array([c[1] for c in colors], dtype=np.uint16),
        "blue": np.array([c[2] for c in colors], dtype=np.uint16)
    }

def make_torus(outer_radius=2.0, inner_radius=0.8, base_z=0.0, material="concrete", classification="unclassified", point_spacing=0.05):
    """Create a torus (donut) shape."""
    points = []
    intensities = []
    classes = []
    colors = []
    
    # Create torus using parametric equations
    n_major = max(20, int(2 * np.pi * outer_radius / point_spacing))
    n_minor = max(15, int(2 * np.pi * inner_radius / point_spacing))
    
    for i in range(n_major):
        u = (i / n_major) * 2 * np.pi
        for j in range(n_minor):
            v = (j / n_minor) * 2 * np.pi
            
            x = (outer_radius + inner_radius * np.cos(v)) * np.cos(u)
            y = (outer_radius + inner_radius * np.cos(v)) * np.sin(u)
            z = base_z + inner_radius * np.sin(v)
            
            points.append([x, y, z])
            intensities.append(MAT_INT[material][0])
            classes.append(CLASS[classification])
            colors.append(MAT_RGB_255[material])
    
    return {
        "x": np.array([p[0] for p in points]),
        "y": np.array([p[1] for p in points]),
        "z": np.array([p[2] for p in points]),
        "intensity": np.array(intensities, dtype=np.uint16),
        "class": np.array(classes, dtype=np.uint8),
        "red": np.array([c[0] for c in colors], dtype=np.uint16),
        "green": np.array([c[1] for c in colors], dtype=np.uint16),
        "blue": np.array([c[2] for c in colors], dtype=np.uint16)
    }

def make_octagon(radius=2.0, base_z=0.0, material="concrete", classification="unclassified", point_spacing=0.05):
    """Create an octagonal shape."""
    points = []
    intensities = []
    classes = []
    colors = []
    
    # Create octagon using 8 triangular sections
    for i in range(8):
        angle1 = i * np.pi / 4
        angle2 = (i + 1) * np.pi / 4
        
        x1 = radius * np.cos(angle1)
        y1 = radius * np.sin(angle1)
        x2 = radius * np.cos(angle2)
        y2 = radius * np.sin(angle2)
        
        # Create triangular section
        for x in np.arange(-radius, radius, point_spacing):
            for y in np.arange(-radius, radius, point_spacing):
                # Check if point is inside this triangular section
                if (x * (y2 - y1) + y * (x1 - x2) + x2 * y1 - x1 * y2) >= 0:
                    # Additional check to ensure it's within the octagon
                    dist = np.sqrt(x**2 + y**2)
                    if dist <= radius:
                        points.append([x, y, base_z])
                        intensities.append(MAT_INT[material][0])
                        classes.append(CLASS[classification])
                        colors.append(MAT_RGB_255[material])
    
    return {
        "x": np.array([p[0] for p in points]),
        "y": np.array([p[1] for p in points]),
        "z": np.array([p[2] for p in points]),
        "intensity": np.array(intensities, dtype=np.uint16),
        "class": np.array(classes, dtype=np.uint8),
        "red": np.array([c[0] for c in colors], dtype=np.uint16),
        "green": np.array([c[1] for c in colors], dtype=np.uint16),
        "blue": np.array([c[2] for c in colors], dtype=np.uint16)
    }

def make_ellipse(major_axis=3.0, minor_axis=2.0, base_z=0.0, material="concrete", classification="unclassified", point_spacing=0.05):
    """Create an elliptical shape."""
    points = []
    intensities = []
    classes = []
    colors = []
    
    # Create ellipse using parametric equations
    for x in np.arange(-major_axis, major_axis, point_spacing):
        for y in np.arange(-minor_axis, minor_axis, point_spacing):
            # Check if point is inside ellipse
            if (x**2 / major_axis**2 + y**2 / minor_axis**2) <= 1:
                points.append([x, y, base_z])
                intensities.append(MAT_INT[material][0])
                classes.append(CLASS[classification])
                colors.append(MAT_RGB_255[material])
    
    return {
        "x": np.array([p[0] for p in points]),
        "y": np.array([p[1] for p in points]),
        "z": np.array([p[2] for p in points]),
        "intensity": np.array(intensities, dtype=np.uint16),
        "class": np.array(classes, dtype=np.uint8),
        "red": np.array([c[0] for c in colors], dtype=np.uint16),
        "green": np.array([c[1] for c in colors], dtype=np.uint16),
        "blue": np.array([c[2] for c in colors], dtype=np.uint16)
    }

def make_helix(radius=1.5, height=4.0, turns=3, base_z=0.0, material="metal", classification="unclassified", point_spacing=0.05):
    """Create a helical structure."""
    points = []
    intensities = []
    classes = []
    colors = []
    
    # Create helix using parametric equations
    n_points = max(100, int(turns * 2 * np.pi * radius / point_spacing))
    
    for i in range(n_points):
        t = (i / n_points) * turns * 2 * np.pi
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = base_z + (i / n_points) * height
        
        points.append([x, y, z])
        intensities.append(MAT_INT[material][0])
        classes.append(CLASS[classification])
        colors.append(MAT_RGB_255[material])
    
    return {
        "x": np.array([p[0] for p in points]),
        "y": np.array([p[1] for p in points]),
        "z": np.array([p[2] for p in points]),
        "intensity": np.array(intensities, dtype=np.uint16),
        "class": np.array(classes, dtype=np.uint8),
        "red": np.array([c[0] for c in colors], dtype=np.uint16),
        "green": np.array([c[1] for c in colors], dtype=np.uint16),
        "blue": np.array([c[2] for c in colors], dtype=np.uint16)
    }

def make_parabolic_dish(radius=2.5, depth=0.8, base_z=0.0, material="metal", classification="unclassified", point_spacing=0.05):
    """Create a parabolic dish shape."""
    points = []
    intensities = []
    classes = []
    colors = []
    
    # Create parabolic surface
    for x in np.arange(-radius, radius, point_spacing):
        for y in np.arange(-radius, radius, point_spacing):
            if x**2 + y**2 <= radius**2:
                # Parabolic equation: z = (x^2 + y^2) / (4 * focal_length)
                focal_length = radius**2 / (4 * depth)
                z = base_z + (x**2 + y**2) / (4 * focal_length)
                
                points.append([x, y, z])
                intensities.append(MAT_INT[material][0])
                classes.append(CLASS[classification])
                colors.append(MAT_RGB_255[material])
    
    return {
        "x": np.array([p[0] for p in points]),
        "y": np.array([p[1] for p in points]),
        "z": np.array([p[2] for p in points]),
        "intensity": np.array(intensities, dtype=np.uint16),
        "class": np.array(classes, dtype=np.uint8),
        "red": np.array([c[0] for c in colors], dtype=np.uint16),
        "green": np.array([c[1] for c in colors], dtype=np.uint16),
        "blue": np.array([c[2] for c in colors], dtype=np.uint16)
    }

def make_geodesic_dome(radius=2.0, base_z=0.0, material="concrete", classification="unclassified", point_spacing=0.05):
    """Create a geodesic dome structure."""
    points = []
    intensities = []
    classes = []
    colors = []
    
    # Create dome using spherical coordinates
    n_theta = max(20, int(np.pi * radius / point_spacing))
    n_phi = max(20, int(2 * np.pi * radius / point_spacing))
    
    for i in range(n_theta):
        theta = (i / n_theta) * np.pi / 2  # 0 to π/2 (hemisphere)
        for j in range(n_phi):
            phi = (j / n_phi) * 2 * np.pi
            
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = base_z + radius * np.cos(theta)
            
            points.append([x, y, z])
            intensities.append(MAT_INT[material][0])
            classes.append(CLASS[classification])
            colors.append(MAT_RGB_255[material])
    
    return {
        "x": np.array([p[0] for p in points]),
        "y": np.array([p[1] for p in points]),
        "z": np.array([p[2] for p in points]),
        "intensity": np.array(intensities, dtype=np.uint16),
        "class": np.array(classes, dtype=np.uint8),
        "red": np.array([c[0] for c in colors], dtype=np.uint16),
        "green": np.array([c[1] for c in colors], dtype=np.uint16),
        "blue": np.array([c[2] for c in colors], dtype=np.uint16)
    }

# -----------------------------
# Geometric Shape Definitions
# -----------------------------

def build_geometric_shapes():
    """Build list of geometric shapes for drawing tool testing."""
    shapes = []
    
    # Basic geometric primitives - 4x larger for easier drawing
    shapes.append(("flat_plane", lambda **kwargs: make_plane(
        kwargs.get('width', 40.0), kwargs.get('length', 40.0), 
        kwargs.get('base_z', 0.0), "concrete", "ground", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("rotated_plane", lambda **kwargs: make_rotated_plane(
        kwargs.get('width', 32.0), kwargs.get('length', 32.0), 
        kwargs.get('base_z', 0.0), "concrete", "ground", kwargs.get('spacing', 0.05), 
        kwargs.get('rotation_angle', 15)
    )))
    
    shapes.append(("stepped_stairs", lambda **kwargs: make_stairs(
        kwargs.get('width', 3.0), kwargs.get('depth', 0.4), kwargs.get('step_h', 0.2),
        kwargs.get('n_steps', 8), kwargs.get('base_z', 0.0), "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    # Complex geometric shapes - slightly larger for better drawing
    shapes.append(("sphere", lambda **kwargs: make_sphere(
        kwargs.get('radius', 2.5), kwargs.get('base_z', 0.0), 
        "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("stepped_pyramid", lambda **kwargs: make_stepped_pyramid(
        kwargs.get('base_size', 4.0), kwargs.get('height', 3.2), kwargs.get('levels', 5),
        kwargs.get('base_z', 0.0), "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("cone", lambda **kwargs: make_cone(
        kwargs.get('radius', 2.8), kwargs.get('height', 4.2), kwargs.get('base_z', 0.0),
        "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("torus", lambda **kwargs: make_torus(
        kwargs.get('outer_radius', 2.6), kwargs.get('inner_radius', 1.3), kwargs.get('base_z', 0.0),
        "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("hexagon", lambda **kwargs: make_hexagon(
        kwargs.get('radius', 2.8), kwargs.get('base_z', 0.0), 
        "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("octagon", lambda **kwargs: make_octagon(
        kwargs.get('radius', 2.6), kwargs.get('base_z', 0.0), 
        "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("ellipse", lambda **kwargs: make_ellipse(
        kwargs.get('major_axis', 4.2), kwargs.get('minor_axis', 3.0), kwargs.get('base_z', 0.0),
        "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("cross", lambda **kwargs: make_cross(
        kwargs.get('length', 3.8), kwargs.get('width', 0.6), kwargs.get('base_z', 0.0),
        "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("helix", lambda **kwargs: make_helix(
        kwargs.get('radius', 2.2), kwargs.get('height', 5.2), kwargs.get('turns', 4), kwargs.get('base_z', 0.0),
        "metal", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("parabolic_dish", lambda **kwargs: make_parabolic_dish(
        kwargs.get('radius', 3.4), kwargs.get('depth', 1.2), kwargs.get('base_z', 0.0),
        "metal", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    shapes.append(("geodesic_dome", lambda **kwargs: make_geodesic_dome(
        kwargs.get('radius', 2.8), kwargs.get('base_z', 0.0), 
        "concrete", "unclassified", kwargs.get('spacing', 0.05)
    )))
    
    return shapes

# -----------------------------
# Density and spacing functions
# -----------------------------

def get_spacing_for_density(density):
    """Calculate grid spacing for target density."""
    return (1.0 / density) ** 0.5

def make_feature_with_density(feature_func, density):
    """Generate a feature with specific density."""
    spacing = get_spacing_for_density(density)
    return feature_func(spacing=spacing)

# -----------------------------
# Main generation function
# -----------------------------

def generate_geometric_shapes_test():
    """Generate the geometric shapes test grid."""
    print("Generating Geometric Shapes Test Grid")
    print("=" * 50)
    
    # Create output directory
    OUT_DIR.mkdir(exist_ok=True)
    
    # Build geometric shapes
    shapes = build_geometric_shapes()
    print(f"Using {len(shapes)} geometric shapes with {len(DENSITY_LEVELS)} density level(s)")
    
    # Generate grid
    all_scenes = []
    grid_metadata = []
    
    print(f"Generating geometric shapes test grid with {len(shapes)} shapes")
    
    for row, (shape_name, shape_func) in enumerate(shapes):
        print(f"Processing row {row+1}/{len(shapes)}: {shape_name}")
        
        for col, density in enumerate(DENSITY_LEVELS):
            x = col * GRID_SPACING
            y = row * GRID_SPACING
            
            # Generate shape with specific density
            try:
                scene_data = make_feature_with_density(shape_func, density)
                scene_data = ensure_cls(scene_data)
                
                # Translate to grid position with Z offset
                scene_data = translate(scene_data, x, y, Z_OFFSET)
                
                all_scenes.append(scene_data)
                
                # Store metadata
                grid_metadata.append({
                    "shape": shape_name,
                    "density": density,
                    "position": (x, y, Z_OFFSET),
                    "points": len(scene_data["x"])
                })
                
            except Exception as e:
                print(f"Error generating {shape_name} at density {density}: {e}")
                continue
    
    # Combine all scenes
    print("Combining all scenes...")
    if not all_scenes:
        print("No scenes generated!")
        return
    
    combined_scene = stack_fields(all_scenes)
    
    # Check point count
    total_points = len(combined_scene["x"])
    print(f"Total points generated: {total_points:,}")
    
    # Save point cloud
    print("Saving point cloud...")
    timestamp = int(time.time())
    output_file = OUT_DIR / f"geometric_shapes_test_{timestamp}.laz"
    
    # Create LAS file
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    
    las.x = combined_scene["x"].astype(np.float64)
    las.y = combined_scene["y"].astype(np.float64)
    las.z = combined_scene["z"].astype(np.float64)
    las.intensity = combined_scene["intensity"].astype(np.uint16)
    las.classification = combined_scene["cls"].astype(np.uint8)
    
    if ADD_RGB and all(k in combined_scene for k in ["red", "green", "blue"]):
        las.red = combined_scene["red"].astype(np.uint16)
        las.green = combined_scene["green"].astype(np.uint16)
        las.blue = combined_scene["blue"].astype(np.uint16)
    
    las.return_number = np.ones(combined_scene["x"].size, dtype=np.uint8)
    las.number_of_returns = np.ones(combined_scene["x"].size, dtype=np.uint8)
    
    if WRITE_LAZ:
        las.write(output_file)
    else:
        las.write(str(output_file).replace('.laz', '.las'))
    
    print(f"Saved: {output_file}")
    print(f"Total points: {total_points:,}")
    
    # Save metadata
    metadata_file = OUT_DIR / f"geometric_shapes_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "density_levels": DENSITY_LEVELS,
            "grid_spacing": GRID_SPACING,
            "shapes": [name for name, _ in shapes],
            "grid_metadata": grid_metadata,
            "total_points": total_points,
            "z_offset": Z_OFFSET
        }, f, indent=2)
    
    print(f"Saved metadata: {metadata_file}")
    
    # Show preview
    if SHOW_PREVIEW and HAVE_O3D:
        print("\nOpening 3D preview...")
        pts = np.vstack([combined_scene["x"], combined_scene["y"], combined_scene["z"]]).T
        
        if ADD_RGB and all(k in combined_scene for k in ["red", "green", "blue"]):
            colors = np.vstack([
                combined_scene["red"],
                combined_scene["green"],
                combined_scene["blue"]
            ]).T.astype(np.float32) / 65535.0
        else:
            colors = (combined_scene["intensity"].astype(np.float32) / 65535.0).reshape(-1, 1)
            colors = np.repeat(colors, 3, axis=1)
        
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Geometric Shapes Test Grid", width=1200, height=800)
        vis.add_geometry(pcd)
        
        # Configure render options
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # Set camera to look at Z offset level
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(0.3)
        
        print("  Enhanced preview with better zoom controls:")
        print("  - Mouse wheel: Zoom in/out")
        print("  - Right-click + drag: Pan")
        print("  - Left-click + drag: Rotate")
        print("  - Close window to continue...")
        
        vis.run()
        vis.destroy_window()
    
    elif SHOW_PREVIEW and not HAVE_O3D:
        print("Open3D not available for preview")
    
    print("\nGeometric shapes test generation complete!")
    print(f"Output directory: {OUT_DIR}")

if __name__ == "__main__":
    generate_geometric_shapes_test()
