#!/usr/bin/env python3
"""
Generate Hollywood sign using the same text approach as density test grid.
This creates human-readable "HOLLYWOOD" letters using point cloud segments.
"""

import numpy as np
import laspy
import json
import time
from pathlib import Path
import zipfile
import sys
sys.path.append('..')
from generate_point_cloud_sandbox import (
    stack_fields, CLASS, MAT_INT, RNG
)
import open3d as o3d

# Configuration
LETTER_DENSITY = 75  # Very low density to clearly show individual points
Z_OFFSET = 0.0

# Output configuration
OUTPUT_DIR = Path("hollywood_text_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_letter_points(letter, x, y, z, width, height, spacing):
    """Create point cloud representation of a letter using segment-based approach."""
    
    # Define segment endpoints (normalized coordinates 0-1)
    # Using a grid-based approach similar to 7-segment display
    segments = {
        # Horizontal segments
        'top': [(0.1, 0.9), (0.9, 0.9)],      # top horizontal
        'mid': [(0.1, 0.5), (0.9, 0.5)],      # middle horizontal  
        'bot': [(0.1, 0.1), (0.9, 0.1)],      # bottom horizontal
        
        # Vertical segments
        'left_top': [(0.1, 0.5), (0.1, 0.9)],    # left vertical top
        'left_bot': [(0.1, 0.1), (0.1, 0.5)],    # left vertical bottom
        'right_top': [(0.9, 0.5), (0.9, 0.9)],   # right vertical top
        'right_bot': [(0.9, 0.1), (0.9, 0.5)],   # right vertical bottom
        
        # Additional segments for better letter shapes
        'left_full': [(0.1, 0.1), (0.1, 0.9)],   # left vertical full
        'right_full': [(0.9, 0.1), (0.9, 0.9)],  # right vertical full
        'center_full': [(0.5, 0.1), (0.5, 0.9)], # center vertical full
        
        # Y-specific segments for proper Y shape
        'y_left_diag': [(0.1, 0.9), (0.5, 0.5)],  # Y left diagonal
        'y_right_diag': [(0.9, 0.9), (0.5, 0.5)], # Y right diagonal
        'y_stem': [(0.5, 0.5), (0.5, 0.1)],       # Y center stem
        
        # W-specific segments - proper W shape (flipped from M)
        'w_left': [(0.1, 0.1), (0.1, 0.9)],      # W left vertical
        'w_right': [(0.9, 0.1), (0.9, 0.9)],     # W right vertical
        'w_diag1': [(0.1, 0.1), (0.3, 0.5)],     # W left diagonal up
        'w_diag2': [(0.3, 0.5), (0.5, 0.9)],     # W center diagonal up
        'w_diag3': [(0.5, 0.9), (0.7, 0.5)],     # W center diagonal down
        'w_diag4': [(0.7, 0.5), (0.9, 0.1)],     # W right diagonal down
        
        # D-specific segments - proper D shape with curve
        'd_left': [(0.1, 0.1), (0.1, 0.9)],      # D left vertical
        'd_top': [(0.1, 0.9), (0.6, 0.9)],       # D top horizontal
        'd_bot': [(0.1, 0.1), (0.6, 0.1)],       # D bottom horizontal
        'd_curve_top': [(0.6, 0.9), (0.8, 0.7)], # D top curve
        'd_curve_mid': [(0.8, 0.7), (0.8, 0.3)], # D middle vertical
        'd_curve_bot': [(0.8, 0.3), (0.6, 0.1)], # D bottom curve
        
        # Diagonal segments
        'diag_lr': [(0.1, 0.9), (0.9, 0.1)],     # diagonal left-to-right
        'diag_rl': [(0.9, 0.9), (0.1, 0.1)],     # diagonal right-to-left
    }

    # Define which segments are lit for each letter
    letter_segments = {
        'H': ['left_full', 'right_full', 'mid'],
        'O': ['top', 'bot', 'left_full', 'right_full'],
        'L': ['left_full', 'bot'],
        'Y': ['y_left_diag', 'y_right_diag', 'y_stem'],
        'W': ['w_left', 'w_right', 'w_diag1', 'w_diag2', 'w_diag3', 'w_diag4'],
        'D': ['d_left', 'd_top', 'd_bot', 'd_curve_top', 'd_curve_mid', 'd_curve_bot'],
    }

    if letter not in letter_segments:
        return None

    points = []
    segment_thickness = 0.25  # Thickness of each segment (25% of character size)

    for segment_name in letter_segments[letter]:
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
                num_points_along = max(5, int(length * width / spacing))
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

def make_hollywood_text_label(text, x, y, z, spacing):
    """Create human-readable text label using point cloud segments with fun colors."""
    
    # Create points to form the text
    all_points = []
    all_colors = []
    all_intensities = []
    all_classes = []
    
    # Character dimensions (appropriately sized for visibility)
    char_width = 6.0
    char_height = 8.0
    char_spacing = 1.5
    
    # Starting position for the text
    start_x = x - (len(text) * (char_width + char_spacing)) / 2
    
    # Define fun colors for each letter (Hollywood-style colors)
    letter_colors = [
        [1.0, 0.0, 0.0],  # H - Red
        [1.0, 0.5, 0.0],  # O - Orange
        [1.0, 1.0, 0.0],  # L - Yellow
        [1.0, 1.0, 0.0],  # L - Yellow
        [0.0, 1.0, 0.0],  # Y - Green
        [0.0, 0.0, 1.0],  # W - Blue
        [0.5, 0.0, 1.0],  # O - Purple
        [0.5, 0.0, 1.0],  # O - Purple
        [1.0, 0.0, 1.0],  # D - Magenta
    ]
    
    for i, letter in enumerate(text):
        char_x = start_x + i * (char_width + char_spacing)
        letter_points = create_letter_points(letter, char_x, y, z, char_width, char_height, spacing)
        
        if letter_points is not None:
            all_points.append(letter_points)
            
            # Create gradient colors for each letter
            num_points = len(letter_points)
            base_color = letter_colors[i % len(letter_colors)]
            
            # Create a gradient from the base color to white
            colors = np.zeros((num_points, 3))
            for j in range(num_points):
                # Create gradient based on Y position (height)
                y_pos = letter_points[j, 1] - (char_x - char_width/2)  # Relative Y position
                gradient_factor = (y_pos + char_height/2) / char_height  # 0 to 1
                gradient_factor = max(0, min(1, gradient_factor))  # Clamp to 0-1
                
                # Interpolate between base color and white
                colors[j] = [
                    base_color[0] + (1.0 - base_color[0]) * gradient_factor,
                    base_color[1] + (1.0 - base_color[1]) * gradient_factor,
                    base_color[2] + (1.0 - base_color[2]) * gradient_factor
                ]
            
            all_colors.append(colors)
            # High intensity for visibility
            intensities = np.full(len(letter_points), 65535, dtype=np.uint16)
            all_intensities.append(intensities)
            # Classification
            classes = np.full(len(letter_points), CLASS["unclassified"], dtype=np.uint8)
            all_classes.append(classes)
    
    if not all_points:
        return None
    
    # Combine all letter points
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
    result["red"] = (combined_colors[:, 0] * 65535).astype(np.uint16)
    result["green"] = (combined_colors[:, 1] * 65535).astype(np.uint16)
    result["blue"] = (combined_colors[:, 2] * 65535).astype(np.uint16)
    
    return result

def create_hollywood_text():
    """Create the Hollywood sign text."""
    print("Creating Hollywood sign text...")
    
    # Calculate spacing for target density
    spacing = (1.0 / LETTER_DENSITY) ** 0.5
    
    # Create the text
    text = "HOLLYWOOD"
    x = 0.0
    y = 0.0
    z = 0.0
    
    scene = make_hollywood_text_label(text, x, y, z, spacing)
    
    return scene

def write_las_file(data, filename):
    """Write point cloud data to LAS file."""
    print(f"Writing {filename}...")
    
    # Ensure all arrays are the same length
    min_len = min(len(data["x"]), len(data["y"]), len(data["z"]))
    for key in data:
        if key in ["x", "y", "z", "red", "green", "blue", "intensity", "class"]:
            data[key] = data[key][:min_len]
    
    # Create LAS file
    las = laspy.create(
        point_format=2,
        file_version="1.4"
    )
    
    # Set header
    las.header.x_scale = 0.001
    las.header.y_scale = 0.001
    las.header.z_scale = 0.001
    las.header.x_offset = np.mean(data["x"])
    las.header.y_offset = np.mean(data["y"])
    las.header.z_offset = np.mean(data["z"])
    
    # Set point data
    las.x = np.asarray(data["x"], dtype=np.float64)
    las.y = np.asarray(data["y"], dtype=np.float64)
    las.z = np.asarray(data["z"], dtype=np.float64)
    las.red = np.asarray(data["red"], dtype=np.uint16)
    las.green = np.asarray(data["green"], dtype=np.uint16)
    las.blue = np.asarray(data["blue"], dtype=np.uint16)
    las.intensity = np.asarray(data["intensity"], dtype=np.uint16)
    las.classification = np.asarray(data["class"], dtype=np.uint8)
    
    # Write file
    las.write(filename)
    print(f"  Wrote {len(data['x']):,} points to {filename}")

def show_preview(las_filename):
    """Show 3D preview of the generated text."""
    print("\nLoading point cloud for preview...")
    
    # Read the LAS file
    las = laspy.read(las_filename)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([las.x, las.y, las.z]))
    
    # Set colors if available
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.column_stack([las.red, las.green, las.blue]) / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"Loaded {len(pcd.points)} points for preview")
    
    # Debug: Print point cloud bounds and colors
    points = np.column_stack([las.x, las.y, las.z])
    print(f"Point cloud bounds:")
    print(f"  X: {points[:, 0].min():.1f} to {points[:, 0].max():.1f}")
    print(f"  Y: {points[:, 1].min():.1f} to {points[:, 1].max():.1f}")
    print(f"  Z: {points[:, 2].min():.1f} to {points[:, 2].max():.1f}")
    
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        print(f"Color ranges:")
        print(f"  Red: {las.red.min()} to {las.red.max()}")
        print(f"  Green: {las.green.min()} to {las.green.max()}")
        print(f"  Blue: {las.blue.min()} to {las.blue.max()}")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Hollywood Sign Text - Segment Based", width=1200, height=800)
    vis.add_geometry(pcd)
    
    # Set up camera to view the text clearly
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    # Look at the center of the point cloud
    center_x = (points[:, 0].min() + points[:, 0].max()) / 2
    center_y = (points[:, 1].min() + points[:, 1].max()) / 2
    center_z = (points[:, 2].min() + points[:, 2].max()) / 2
    ctr.set_lookat([center_x, center_y, center_z])
    ctr.set_zoom(0.3)
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = 1.5  # Even smaller point size to show individual points clearly
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Very dark background for better contrast
    render_option.show_coordinate_frame = True  # Show coordinate frame for reference
    
    print("Showing 3D preview...")
    print("Close the preview window to continue.")
    vis.run()
    vis.destroy_window()

def main():
    """Main function to generate the Hollywood sign text."""
    print("Generating Hollywood Sign Text - Segment Based")
    print("=" * 60)
    
    # Generate text
    scene = create_hollywood_text()
    
    # Generate timestamp
    timestamp = int(time.time())
    
    # Write main scene file
    las_filename = OUTPUT_DIR / f"hollywood_text_{timestamp}.laz"
    write_las_file(scene, las_filename)
    
    # Create metadata
    metadata = {
        "scene_type": "hollywood_text",
        "timestamp": timestamp,
        "total_points": len(scene["x"]),
        "letter_density": LETTER_DENSITY,
        "z_offset": Z_OFFSET,
        "description": "Hollywood sign text using segment-based approach like density test grid"
    }
    
    metadata_filename = OUTPUT_DIR / f"hollywood_text_metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create zip file
    zip_filename = OUTPUT_DIR / f"hollywood_text_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(las_filename, las_filename.name)
        zipf.write(metadata_filename, metadata_filename.name)
    
    print(f"\nScene generation complete!")
    print(f"Total points: {len(scene['x']):,}")
    print(f"Files created:")
    print(f"  - {las_filename}")
    print(f"  - {metadata_filename}")
    print(f"  - {zip_filename}")
    
    # Show preview before cleaning up
    show_preview(las_filename)
    
    # Clean up individual files
    las_filename.unlink()
    metadata_filename.unlink()

if __name__ == "__main__":
    main()
