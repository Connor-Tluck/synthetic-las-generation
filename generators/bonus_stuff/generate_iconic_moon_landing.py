#!/usr/bin/env python3
"""
Generate an iconic moon landing scene point cloud based on the famous Apollo 11 photograph.
This creates a scene with an astronaut, American flag, and detailed lunar surface.
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
    make_plane, make_box, make_cylinder, make_sphere,
    stack_fields, CLASS, MAT_INT, RNG
)
import open3d as o3d

# Configuration
MOON_DENSITY = 3000  # High density for detailed surface
OBJECT_DENSITY = 6000  # Very high density for objects
DETAIL_DENSITY = 8000  # Highest density for fine details
Z_OFFSET = 0.0  # No Z offset for this scene

# Scene dimensions (smaller scale for iconic composition)
SCENE_WIDTH = 20.0
SCENE_HEIGHT = 15.0
MOON_SURFACE_SIZE = 25.0

# Output configuration
OUTPUT_DIR = Path("iconic_moon_landing_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def make_detailed_moon_surface(width=25.0, height=25.0, density=3000):
    """Create a detailed lunar surface with craters, rocks, and varied colors."""
    print("  Creating detailed moon surface...")
    
    # Calculate spacing from density
    spacing = np.sqrt(1.0 / density)
    
    # Main surface plane
    surface = make_plane(width, height, 0.0, "gravel", "ground", spacing)
    
    # Add craters of various sizes
    crater_positions = [
        (8.0, 5.0, 0.1), (15.0, 8.0, 0.08), (3.0, 12.0, 0.12),
        (18.0, 3.0, 0.06), (6.0, 18.0, 0.15), (20.0, 15.0, 0.1)
    ]
    
    for x, y, radius in crater_positions:
        crater = make_sphere(radius, -radius * 0.7, "gravel", "ground", spacing)
        crater["x"] += x
        crater["y"] += y
        crater["red"] = np.clip(crater["red"] * 0.8, 0, 255).astype(np.uint8)
        crater["green"] = np.clip(crater["green"] * 0.7, 0, 255).astype(np.uint8)
        crater["blue"] = np.clip(crater["blue"] * 0.6, 0, 255).astype(np.uint8)
        surface = stack_fields([surface, crater])
    
    # Add scattered rocks
    for _ in range(25):
        rock_size = RNG.uniform(0.05, 0.2)
        rock = make_sphere(rock_size, RNG.uniform(0.01, 0.1), "gravel", "ground", spacing)
        rock["x"] += RNG.uniform(0, width)
        rock["y"] += RNG.uniform(0, height)
        rock["red"] = np.clip(rock["red"] * 0.9, 0, 255).astype(np.uint8)
        rock["green"] = np.clip(rock["green"] * 0.8, 0, 255).astype(np.uint8)
        rock["blue"] = np.clip(rock["blue"] * 0.7, 0, 255).astype(np.uint8)
        surface = stack_fields([surface, rock])
    
    # Add surface texture variations
    for _ in range(50):
        texture_size = RNG.uniform(0.1, 0.3)
        texture = make_sphere(texture_size, RNG.uniform(-0.05, 0.05), "gravel", "ground", spacing)
        texture["x"] += RNG.uniform(0, width)
        texture["y"] += RNG.uniform(0, height)
        texture["red"] = np.clip(texture["red"] * RNG.uniform(0.7, 1.2), 0, 255).astype(np.uint8)
        texture["green"] = np.clip(texture["green"] * RNG.uniform(0.6, 1.1), 0, 255).astype(np.uint8)
        texture["blue"] = np.clip(texture["blue"] * RNG.uniform(0.5, 1.0), 0, 255).astype(np.uint8)
        surface = stack_fields([surface, texture])
    
    return surface

def make_astronaut_figure(x=0, y=0, z=0):
    """Create a detailed astronaut figure."""
    print("  Creating astronaut figure...")
    
    # Calculate spacing from density
    spacing = np.sqrt(1.0 / OBJECT_DENSITY)
    
    # Main body (torso)
    torso = make_box(0.8, 0.6, 1.2, z + 0.6, "metal", "building", spacing)
    torso["x"] += x
    torso["y"] += y
    torso["red"] = np.full(len(torso["x"]), 240, dtype=np.uint16)
    torso["green"] = np.full(len(torso["x"]), 240, dtype=np.uint16)
    torso["blue"] = np.full(len(torso["x"]), 240, dtype=np.uint16)
    
    # Helmet
    helmet = make_sphere(0.4, z + 1.4, "metal", "building", spacing)
    helmet["x"] += x
    helmet["y"] += y
    helmet["red"] = np.full(len(helmet["x"]), 250, dtype=np.uint16)
    helmet["green"] = np.full(len(helmet["x"]), 250, dtype=np.uint16)
    helmet["blue"] = np.full(len(helmet["x"]), 250, dtype=np.uint16)
    
    # Arms
    left_arm = make_cylinder(0.15, 0.8, z + 0.8, "metal", "building", spacing)
    left_arm["x"] += x - 0.5
    left_arm["y"] += y
    left_arm["red"] = np.full(len(left_arm["x"]), 240, dtype=np.uint16)
    left_arm["green"] = np.full(len(left_arm["x"]), 240, dtype=np.uint16)
    left_arm["blue"] = np.full(len(left_arm["x"]), 240, dtype=np.uint16)
    
    right_arm = make_cylinder(0.15, 0.8, z + 0.8, "metal", "building", spacing)
    right_arm["x"] += x + 0.5
    right_arm["y"] += y
    right_arm["red"] = np.full(len(right_arm["x"]), 240, dtype=np.uint16)
    right_arm["green"] = np.full(len(right_arm["x"]), 240, dtype=np.uint16)
    right_arm["blue"] = np.full(len(right_arm["x"]), 240, dtype=np.uint16)
    
    # Legs
    left_leg = make_cylinder(0.2, 1.0, z, "metal", "building", spacing)
    left_leg["x"] += x - 0.2
    left_leg["y"] += y
    left_leg["red"] = np.full(len(left_leg["x"]), 240, dtype=np.uint16)
    left_leg["green"] = np.full(len(left_leg["x"]), 240, dtype=np.uint16)
    left_leg["blue"] = np.full(len(left_leg["x"]), 240, dtype=np.uint16)
    
    right_leg = make_cylinder(0.2, 1.0, z, "metal", "building", spacing)
    right_leg["x"] += x + 0.2
    right_leg["y"] += y
    right_leg["red"] = np.full(len(right_leg["x"]), 240, dtype=np.uint16)
    right_leg["green"] = np.full(len(right_leg["x"]), 240, dtype=np.uint16)
    right_leg["blue"] = np.full(len(right_leg["x"]), 240, dtype=np.uint16)
    
    # Boots
    left_boot = make_box(0.3, 0.4, 0.2, z - 0.1, "metal", "building", spacing)
    left_boot["x"] += x - 0.2
    left_boot["y"] += y
    left_boot["red"] = np.full(len(left_boot["x"]), 220, dtype=np.uint16)
    left_boot["green"] = np.full(len(left_boot["x"]), 220, dtype=np.uint16)
    left_boot["blue"] = np.full(len(left_boot["x"]), 220, dtype=np.uint16)
    
    right_boot = make_box(0.3, 0.4, 0.2, z - 0.1, "metal", "building", spacing)
    right_boot["x"] += x + 0.2
    right_boot["y"] += y
    right_boot["red"] = np.full(len(right_boot["x"]), 220, dtype=np.uint16)
    right_boot["green"] = np.full(len(right_boot["x"]), 220, dtype=np.uint16)
    right_boot["blue"] = np.full(len(right_boot["x"]), 220, dtype=np.uint16)
    
    # Backpack (PLSS)
    backpack = make_box(0.6, 0.4, 0.8, z + 0.6, "metal", "building", spacing)
    backpack["x"] += x
    backpack["y"] += y - 0.3
    backpack["red"] = np.full(len(backpack["x"]), 230, dtype=np.uint16)
    backpack["green"] = np.full(len(backpack["x"]), 230, dtype=np.uint16)
    backpack["blue"] = np.full(len(backpack["x"]), 230, dtype=np.uint16)
    
    # Antenna on backpack
    antenna_spacing = np.sqrt(1.0 / DETAIL_DENSITY)
    antenna = make_cylinder(0.02, 0.3, z + 1.0, "metal", "building", antenna_spacing)
    antenna["x"] += x
    antenna["y"] += y - 0.3
    antenna["red"] = np.full(len(antenna["x"]), 200, dtype=np.uint16)
    antenna["green"] = np.full(len(antenna["x"]), 200, dtype=np.uint16)
    antenna["blue"] = np.full(len(antenna["x"]), 200, dtype=np.uint16)
    
    # Combine all parts
    print("  Combining astronaut parts...")
    print(f"  Torso red shape: {torso['red'].shape}")
    print(f"  Helmet red shape: {helmet['red'].shape}")
    astronaut = stack_fields([torso, helmet])
    print(f"  Torso + helmet: {len(astronaut['x'])} points")
    astronaut = stack_fields([astronaut, left_arm])
    print(f"  + left_arm: {len(astronaut['x'])} points")
    astronaut = stack_fields([astronaut, right_arm])
    print(f"  + right_arm: {len(astronaut['x'])} points")
    astronaut = stack_fields([astronaut, left_leg])
    print(f"  + left_leg: {len(astronaut['x'])} points")
    astronaut = stack_fields([astronaut, right_leg])
    print(f"  + right_leg: {len(astronaut['x'])} points")
    astronaut = stack_fields([astronaut, left_boot])
    print(f"  + left_boot: {len(astronaut['x'])} points")
    astronaut = stack_fields([astronaut, right_boot])
    print(f"  + right_boot: {len(astronaut['x'])} points")
    astronaut = stack_fields([astronaut, backpack])
    print(f"  + backpack: {len(astronaut['x'])} points")
    astronaut = stack_fields([astronaut, antenna])
    print(f"  + antenna: {len(astronaut['x'])} points")
    
    return astronaut

def make_american_flag(x=0, y=0, z=0):
    """Create an American flag with rippled fabric effect."""
    print("  Creating American flag...")
    
    # Calculate spacing from density
    spacing = np.sqrt(1.0 / OBJECT_DENSITY)
    
    # Flag pole
    pole = make_cylinder(0.02, 2.5, z + 1.25, "metal", "building", spacing)
    pole["x"] += x
    pole["y"] += y
    pole["red"] = np.full(len(pole["x"]), 100, dtype=np.uint16)
    pole["green"] = np.full(len(pole["x"]), 100, dtype=np.uint16)
    pole["blue"] = np.full(len(pole["x"]), 100, dtype=np.uint16)
    
    # Flag fabric with ripples
    flag_width = 1.5
    flag_height = 1.0
    
    # Create flag in sections to simulate rippling
    for i in range(8):
        section_width = flag_width / 8
        section_height = flag_height
        
        # Add wave effect
        wave_offset = np.sin(i * 0.5) * 0.1
        
        flag_section = make_plane(section_width, section_height, z + 1.5, "plastic", "building", spacing)
        flag_section["x"] += x + i * section_width + 0.1
        flag_section["y"] += y + wave_offset
        
        # Color the flag sections
        if i < 2:  # Blue field
            flag_section["red"] = np.full(len(flag_section["x"]), 0, dtype=np.uint16)
            flag_section["green"] = np.full(len(flag_section["x"]), 50, dtype=np.uint16)
            flag_section["blue"] = np.full(len(flag_section["x"]), 150, dtype=np.uint16)
        else:  # Red and white stripes
            if (i - 2) % 2 == 0:  # Red stripes
                flag_section["red"] = np.full(len(flag_section["x"]), 200, dtype=np.uint16)
                flag_section["green"] = np.full(len(flag_section["x"]), 0, dtype=np.uint16)
                flag_section["blue"] = np.full(len(flag_section["x"]), 0, dtype=np.uint16)
            else:  # White stripes
                flag_section["red"] = np.full(len(flag_section["x"]), 250, dtype=np.uint16)
                flag_section["green"] = np.full(len(flag_section["x"]), 250, dtype=np.uint16)
                flag_section["blue"] = np.full(len(flag_section["x"]), 250, dtype=np.uint16)
        
        pole = stack_fields([pole, flag_section])
    
    return pole

def make_footprints(x_start, y_start, num_steps=8, step_size=0.3):
    """Create a trail of footprints."""
    print("  Creating footprint trail...")
    
    # Calculate spacing from density
    spacing = np.sqrt(1.0 / (DETAIL_DENSITY//2))
    
    footprints = None
    
    for i in range(num_steps):
        # Alternate left and right footprints
        foot_offset = 0.15 if i % 2 == 0 else -0.15
        
        # Create footprint depression
        footprint = make_plane(0.25, 0.15, -0.02, "gravel", "ground", spacing)
        footprint["x"] += x_start + i * step_size
        footprint["y"] += y_start + foot_offset
        footprint["red"] = np.full(len(footprint["x"]), 80, dtype=np.uint16)
        footprint["green"] = np.full(len(footprint["x"]), 80, dtype=np.uint16)
        footprint["blue"] = np.full(len(footprint["x"]), 80, dtype=np.uint16)
        
        if footprints is None:
            footprints = footprint
        else:
            footprints = stack_fields([footprints, footprint])
    
    return footprints

def make_lander_leg(x=0, y=0, z=0):
    """Create a single lander leg (partially visible in iconic image)."""
    print("  Creating lander leg...")
    
    # Calculate spacing from density
    spacing = np.sqrt(1.0 / OBJECT_DENSITY)
    
    # Main leg structure
    leg = make_cylinder(0.1, 1.5, z + 0.75, "metal", "building", spacing)
    leg["x"] += x
    leg["y"] += y
    leg["red"] = np.full(len(leg["x"]), 180, dtype=np.uint16)
    leg["green"] = np.full(len(leg["x"]), 180, dtype=np.uint16)
    leg["blue"] = np.full(len(leg["x"]), 180, dtype=np.uint16)
    
    # Foot pad
    foot_pad = make_cylinder(0.3, 0.1, z, "metal", "building", spacing)
    foot_pad["x"] += x
    foot_pad["y"] += y
    foot_pad["red"] = np.full(len(foot_pad["x"]), 160, dtype=np.uint16)
    foot_pad["green"] = np.full(len(foot_pad["x"]), 160, dtype=np.uint16)
    foot_pad["blue"] = np.full(len(foot_pad["x"]), 160, dtype=np.uint16)
    
    return stack_fields([leg, foot_pad])

def create_iconic_moon_landing_scene():
    """Create the iconic moon landing scene."""
    print("Creating iconic moon landing scene...")
    
    # Moon surface
    moon_surface = make_detailed_moon_surface(MOON_SURFACE_SIZE, MOON_SURFACE_SIZE, MOON_DENSITY)
    moon_surface["z"] += Z_OFFSET
    
    # Astronaut positioned like in the iconic image
    astronaut = make_astronaut_figure(10.0, 8.0, Z_OFFSET)
    
    # American flag positioned to the left of astronaut
    flag = make_american_flag(7.0, 8.0, Z_OFFSET)
    
    # Footprint trail leading to flag
    footprints = make_footprints(12.0, 6.0, 8, 0.4)
    footprints["z"] += Z_OFFSET
    
    # Lander leg (partially visible in background)
    lander_leg = make_lander_leg(2.0, 12.0, Z_OFFSET)
    
    # Combine all elements
    scene = stack_fields([moon_surface, astronaut])
    scene = stack_fields([scene, flag])
    scene = stack_fields([scene, footprints])
    scene = stack_fields([scene, lander_leg])
    
    return scene

def write_las_file(data, filename):
    """Write point cloud data to LAS file."""
    print(f"Writing {filename}...")
    
    # Ensure all arrays are the same length
    min_len = min(len(data["x"]), len(data["y"]), len(data["z"]))
    for key in data:
        if key in ["x", "y", "z", "red", "green", "blue", "intensity", "cls"]:
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
    las.classification = np.asarray(data["cls"], dtype=np.uint8)
    
    # Write file
    las.write(filename)
    print(f"  Wrote {len(data['x']):,} points to {filename}")

def main():
    """Main function to generate the iconic moon landing scene."""
    print("Generating Iconic Moon Landing Scene")
    print("=" * 50)
    
    # Generate scene
    scene = create_iconic_moon_landing_scene()
    
    # Generate timestamp
    timestamp = int(time.time())
    
    # Write main scene file
    las_filename = OUTPUT_DIR / f"iconic_moon_landing_{timestamp}.laz"
    write_las_file(scene, las_filename)
    
    # Create metadata
    metadata = {
        "scene_type": "iconic_moon_landing",
        "timestamp": timestamp,
        "total_points": len(scene["x"]),
        "moon_density": MOON_DENSITY,
        "object_density": OBJECT_DENSITY,
        "detail_density": DETAIL_DENSITY,
        "z_offset": Z_OFFSET,
        "scene_width": SCENE_WIDTH,
        "scene_height": SCENE_HEIGHT,
        "moon_surface_size": MOON_SURFACE_SIZE,
        "description": "Iconic moon landing scene based on famous Apollo 11 photograph"
    }
    
    metadata_filename = OUTPUT_DIR / f"iconic_moon_landing_metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create zip file
    zip_filename = OUTPUT_DIR / f"iconic_moon_landing_{timestamp}.zip"
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

def show_preview(las_filename):
    """Show 3D preview of the generated scene."""
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
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Iconic Moon Landing Scene", width=1200, height=800)
    vis.add_geometry(pcd)
    
    # Set up camera
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    ctr.set_lookat([12.5, 12.5, 0])  # Look at center of scene
    ctr.set_zoom(0.3)
    
    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.2])  # Dark blue background like space
    
    print("Showing 3D preview...")
    print("Close the preview window to continue.")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
