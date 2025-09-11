#!/usr/bin/env python3
"""
Moon Landing Scene Generator
Creates a point cloud representation of the iconic Apollo 11 moon landing scene.
Includes: Moon surface, lunar lander, astronaut, American flag, and video camera.
"""

import numpy as np
import laspy
import open3d as o3d
from datetime import datetime
import os
import json
import zipfile

# Import our utility functions
from generate_point_cloud_sandbox import (
    translate, stack_fields, ensure_cls, 
    MAT_INT, MAT_RGB_255, CLASS
)

# Scene Configuration
SCENE_NAME = "moon_landing_scene"
OUTPUT_DIR = "moon_landing_output"
MAX_POINTS = 50_000_000

# Moon Landing Scene Scale (compact but recognizable)
MOON_SURFACE_SIZE = 30.0  # 30m x 30m moon surface
LANDER_SCALE = 0.8        # Scale factor for lander
ASTRONAUT_SCALE = 0.6     # Scale factor for astronaut
FLAG_SCALE = 0.7          # Scale factor for flag
CAMERA_SCALE = 0.5        # Scale factor for camera

# Point density for different elements
MOON_DENSITY = 2000       # pts/m² for moon surface
OBJECT_DENSITY = 5000     # pts/m² for objects
DETAIL_DENSITY = 8000     # pts/m² for detailed objects

def make_moon_surface(size=30.0, density=2000):
    """Create a cratered moon surface with realistic terrain and varied colors."""
    spacing = 1.0 / np.sqrt(density)
    x = np.arange(-size/2, size/2, spacing)
    y = np.arange(-size/2, size/2, spacing)
    X, Y = np.meshgrid(x, y)
    
    # Create cratered terrain using multiple noise layers
    Z = np.zeros_like(X)
    
    # Large craters
    crater1_x, crater1_y = -8, 5
    crater1_r = 4.0
    dist1 = np.sqrt((X - crater1_x)**2 + (Y - crater1_y)**2)
    Z += -0.8 * np.exp(-(dist1**2) / (2 * crater1_r**2))
    
    crater2_x, crater2_y = 6, -3
    crater2_r = 3.0
    dist2 = np.sqrt((X - crater2_x)**2 + (Y - crater2_y)**2)
    Z += -0.6 * np.exp(-(dist2**2) / (2 * crater2_r**2))
    
    crater3_x, crater3_y = -2, -8
    crater3_r = 2.5
    dist3 = np.sqrt((X - crater3_x)**2 + (Y - crater3_y)**2)
    Z += -0.4 * np.exp(-(dist3**2) / (2 * crater3_r**2))
    
    # Small craters and surface roughness
    for i in range(20):
        cx = np.random.uniform(-size/2, size/2)
        cy = np.random.uniform(-size/2, size/2)
        cr = np.random.uniform(0.5, 1.5)
        depth = np.random.uniform(-0.2, -0.05)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        Z += depth * np.exp(-(dist**2) / (2 * cr**2))
    
    # General surface roughness
    Z += 0.1 * np.sin(X * 0.5) * np.cos(Y * 0.3)
    Z += 0.05 * np.sin(X * 1.2) * np.sin(Y * 0.8)
    
    # Flatten points
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Enhanced moon surface colors with variation
    colors = np.zeros((len(points), 3))
    for i, (px, py, pz) in enumerate(points):
        # Base moon color (grayish-white)
        base_r, base_g, base_b = 180, 180, 190
        
        # Add color variation based on position and height
        # Darker areas in craters
        if pz < -0.3:
            base_r, base_g, base_b = 140, 140, 150  # Darker gray
        elif pz < -0.1:
            base_r, base_g, base_b = 160, 160, 170  # Medium gray
        
        # Add subtle brown tints in some areas
        brown_factor = 0.3 * np.sin(px * 0.1) * np.cos(py * 0.1)
        if brown_factor > 0.1:
            base_r += 20
            base_g += 10
            base_b -= 10
        
        # Add slight blue tint in shadowed areas
        if px < 0 and py > 0:  # Shadow quadrant
            base_b += 15
        
        # Random variation for realistic regolith appearance
        variation = np.random.uniform(-15, 15)
        base_r = max(120, min(220, base_r + variation))
        base_g = max(120, min(220, base_g + variation))
        base_b = max(120, min(220, base_b + variation))
        
        colors[i] = [base_r, base_g, base_b]
    
    return {
        "x": points[:, 0],
        "y": points[:, 1], 
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 150),
        "cls": np.full(len(points), CLASS["ground"])
    }

def make_lunar_lander(scale=0.8, density=5000):
    """Create Apollo-style lunar lander with enhanced details."""
    spacing = 1.0 / np.sqrt(density)
    
    # Main descent stage (cylindrical)
    descent_height = 1.2 * scale
    descent_radius = 1.0 * scale
    descent_points = []
    
    # Descent stage body
    for z in np.arange(0, descent_height, spacing):
        for angle in np.arange(0, 2*np.pi, spacing/descent_radius):
            x = descent_radius * np.cos(angle)
            y = descent_radius * np.sin(angle)
            descent_points.append([x, y, z])
    
    # Landing legs (4 legs)
    leg_points = []
    leg_positions = [
        (1.5*scale, 0, 0), (-1.5*scale, 0, 0),
        (0, 1.5*scale, 0), (0, -1.5*scale, 0)
    ]
    
    for leg_x, leg_y, leg_z in leg_positions:
        # Leg strut
        for t in np.arange(0, 1, spacing):
            x = leg_x * t
            y = leg_y * t
            z = -1.5 * scale * t
            leg_points.append([x, y, z])
        
        # Foot pad
        foot_radius = 0.3 * scale
        for angle in np.arange(0, 2*np.pi, spacing/foot_radius):
            x = leg_x + foot_radius * np.cos(angle)
            y = leg_y + foot_radius * np.sin(angle)
            z = -1.5 * scale
            leg_points.append([x, y, z])
    
    # Ascent stage (smaller cylinder on top)
    ascent_points = []
    ascent_height = 0.8 * scale
    ascent_radius = 0.6 * scale
    for z in np.arange(descent_height, descent_height + ascent_height, spacing):
        for angle in np.arange(0, 2*np.pi, spacing/ascent_radius):
            x = ascent_radius * np.cos(angle)
            y = ascent_radius * np.sin(angle)
            ascent_points.append([x, y, z])
    
    # Add antennas and details
    detail_points = []
    
    # Main antenna on ascent stage
    antenna_height = 0.4 * scale
    for t in np.arange(0, 1, spacing):
        x = 0
        y = 0
        z = descent_height + ascent_height + antenna_height * t
        detail_points.append([x, y, z])
    
    # Side antennas
    for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        for t in np.arange(0, 1, spacing):
            x = ascent_radius * np.cos(angle) + 0.1*scale * np.cos(angle) * t
            y = ascent_radius * np.sin(angle) + 0.1*scale * np.sin(angle) * t
            z = descent_height + ascent_height/2 + 0.2*scale * t
            detail_points.append([x, y, z])
    
    # Thrusters on descent stage
    thruster_positions = [
        (0.8*scale, 0, 0.2*scale), (-0.8*scale, 0, 0.2*scale),
        (0, 0.8*scale, 0.2*scale), (0, -0.8*scale, 0.2*scale)
    ]
    
    for thr_x, thr_y, thr_z in thruster_positions:
        thruster_radius = 0.1 * scale
        for angle in np.arange(0, 2*np.pi, spacing/thruster_radius):
            x = thr_x + thruster_radius * np.cos(angle)
            y = thr_y + thruster_radius * np.sin(angle)
            z = thr_z
            detail_points.append([x, y, z])
    
    # Combine all lander parts
    all_points = descent_points + leg_points + ascent_points + detail_points
    points = np.array(all_points)
    
    # Enhanced lander colors with variation
    colors = np.zeros((len(points), 3))
    for i, (px, py, pz) in enumerate(points):
        if i < len(descent_points):
            # Descent stage - white with slight gray
            colors[i] = [220, 220, 230]
        elif i < len(descent_points) + len(leg_points):
            # Landing legs - slightly darker
            colors[i] = [200, 200, 210]
        elif i < len(descent_points) + len(leg_points) + len(ascent_points):
            # Ascent stage - bright white
            colors[i] = [240, 240, 250]
        else:
            # Details - darker for contrast
            colors[i] = [180, 180, 190]
    
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 200),
        "cls": np.full(len(points), CLASS["building"])
    }

def make_astronaut(scale=0.6, density=5000):
    """Create astronaut figure in spacesuit."""
    spacing = 1.0 / np.sqrt(density)
    points = []
    
    # Helmet (sphere)
    helmet_radius = 0.3 * scale
    for phi in np.arange(0, np.pi, spacing/helmet_radius):
        for theta in np.arange(0, 2*np.pi, spacing/helmet_radius):
            x = helmet_radius * np.sin(phi) * np.cos(theta)
            y = helmet_radius * np.sin(phi) * np.sin(theta)
            z = 1.2 * scale + helmet_radius * np.cos(phi)
            points.append([x, y, z])
    
    # Torso (cylinder)
    torso_height = 0.6 * scale
    torso_radius = 0.25 * scale
    for z in np.arange(0.4*scale, 0.4*scale + torso_height, spacing):
        for angle in np.arange(0, 2*np.pi, spacing/torso_radius):
            x = torso_radius * np.cos(angle)
            y = torso_radius * np.sin(angle)
            points.append([x, y, z])
    
    # Arms
    arm_positions = [(0.4*scale, 0, 0.7*scale), (-0.4*scale, 0, 0.7*scale)]
    for arm_x, arm_y, arm_z in arm_positions:
        for t in np.arange(0, 1, spacing):
            x = arm_x + 0.3*scale * t
            y = arm_y
            z = arm_z - 0.2*scale * t
            points.append([x, y, z])
    
    # Legs
    leg_positions = [(0.15*scale, 0, 0.4*scale), (-0.15*scale, 0, 0.4*scale)]
    for leg_x, leg_y, leg_z in leg_positions:
        for t in np.arange(0, 1, spacing):
            x = leg_x
            y = leg_y
            z = leg_z - 0.5*scale * t
            points.append([x, y, z])
    
    # Boots
    for leg_x, leg_y, leg_z in leg_positions:
        boot_radius = 0.1 * scale
        for angle in np.arange(0, 2*np.pi, spacing/boot_radius):
            x = leg_x + boot_radius * np.cos(angle)
            y = leg_y + boot_radius * np.sin(angle)
            z = leg_z - 0.5*scale
            points.append([x, y, z])
    
    points = np.array(points)
    
    # Astronaut color (white spacesuit)
    colors = np.full((len(points), 3), [240, 240, 250])
    
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 180),
        "cls": np.full(len(points), CLASS["building"])
    }

def make_american_flag(scale=0.7, density=5000):
    """Create American flag on pole."""
    spacing = 1.0 / np.sqrt(density)
    points = []
    
    # Flag pole
    pole_height = 2.0 * scale
    pole_radius = 0.02 * scale
    for z in np.arange(0, pole_height, spacing):
        for angle in np.arange(0, 2*np.pi, spacing/pole_radius):
            x = pole_radius * np.cos(angle)
            y = pole_radius * np.sin(angle)
            points.append([x, y, z])
    
    # Flag fabric (rippled effect)
    flag_width = 0.8 * scale
    flag_height = 0.5 * scale
    flag_x = 0.1 * scale  # Offset from pole
    
    for u in np.arange(0, flag_width, spacing):
        for v in np.arange(0, flag_height, spacing):
            # Rippled flag effect
            ripple = 0.05 * scale * np.sin(u * 10) * np.cos(v * 8)
            x = flag_x + u
            y = ripple
            z = pole_height - 0.1*scale - v
            points.append([x, y, z])
    
    points = np.array(points)
    
    # Flag colors (red, white, blue stripes)
    colors = np.zeros((len(points), 3))
    for i, (x, y, z) in enumerate(points):
        if z > pole_height - 0.1*scale:  # Flag fabric
            # Simple stripe pattern
            stripe_height = flag_height / 13  # 13 stripes
            stripe_num = int((z - (pole_height - 0.1*scale - flag_height)) / stripe_height)
            if stripe_num % 2 == 0:  # Red stripes
                colors[i] = [200, 0, 0]
            else:  # White stripes
                colors[i] = [255, 255, 255]
        else:  # Pole
            colors[i] = [139, 69, 19]  # Brown
    
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 160),
        "cls": np.full(len(points), CLASS["building"])
    }

def make_equipment_bag(scale=0.3, density=4000):
    """Create equipment bag/tool kit."""
    spacing = 1.0 / np.sqrt(density)
    points = []
    
    # Main bag body (rectangular)
    bag_width = 0.4 * scale
    bag_height = 0.3 * scale
    bag_depth = 0.2 * scale
    
    for x in np.arange(-bag_width/2, bag_width/2, spacing):
        for y in np.arange(-bag_depth/2, bag_depth/2, spacing):
            for z in np.arange(0, bag_height, spacing):
                points.append([x, y, z])
    
    # Bag straps
    for t in np.arange(0, 1, spacing):
        # Left strap
        points.append([-bag_width/2 - 0.05*scale, -bag_depth/2 + 0.1*scale*t, bag_height/2])
        # Right strap
        points.append([bag_width/2 + 0.05*scale, -bag_depth/2 + 0.1*scale*t, bag_height/2])
    
    points = np.array(points)
    
    # Equipment bag color (dark green/olive)
    colors = np.full((len(points), 3), [80, 100, 60])
    
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 120),
        "cls": np.full(len(points), CLASS["building"])
    }

def make_scattered_rocks(scale=1.0, density=2000):
    """Create scattered rocks and debris around the landing site."""
    spacing = 1.0 / np.sqrt(density)
    points = []
    
    # Rock positions around the landing site
    rock_positions = [
        (2.0, -1.5, 0.05), (-3.0, 2.0, 0.03), (4.5, 1.0, 0.04),
        (-2.0, -2.5, 0.06), (1.5, 4.0, 0.02), (-4.0, -1.0, 0.05),
        (3.5, -3.0, 0.03), (-1.0, 3.5, 0.04), (0.5, -4.0, 0.02)
    ]
    
    for rx, ry, rz in rock_positions:
        # Create irregular rock shape
        rock_size = np.random.uniform(0.1, 0.3) * scale
        for i in range(int(50 * rock_size)):
            # Random points within rock volume
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0, rock_size)
            height = np.random.uniform(0, rock_size * 0.6)
            
            x = rx + radius * np.cos(angle)
            y = ry + radius * np.sin(angle)
            z = rz + height
            points.append([x, y, z])
    
    points = np.array(points)
    
    # Rock colors (various grays and browns)
    colors = np.zeros((len(points), 3))
    for i in range(len(points)):
        # Random rock color variation
        base_gray = np.random.uniform(100, 160)
        colors[i] = [base_gray, base_gray, base_gray + np.random.uniform(-10, 10)]
    
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 80),
        "cls": np.full(len(points), CLASS["ground"])
    }

def make_lander_ladder(scale=0.8, density=5000):
    """Create ladder from lander to surface."""
    spacing = 1.0 / np.sqrt(density)
    points = []
    
    # Ladder rails (2 vertical rails)
    rail_positions = [(-0.3*scale, 0, 0), (0.3*scale, 0, 0)]
    
    for rail_x, rail_y, rail_z in rail_positions:
        for t in np.arange(0, 1, spacing):
            x = rail_x
            y = rail_y
            z = 0.5*scale + 1.0*scale * t  # From surface to lander
            points.append([x, y, z])
    
    # Ladder rungs (horizontal steps)
    for rung_z in np.arange(0.5*scale, 1.5*scale, 0.15*scale):
        for t in np.arange(0, 1, spacing):
            x = -0.3*scale + 0.6*scale * t
            y = 0
            z = rung_z
            points.append([x, y, z])
    
    points = np.array(points)
    
    # Ladder color (metallic)
    colors = np.full((len(points), 3), [180, 180, 190])
    
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 180),
        "cls": np.full(len(points), CLASS["building"])
    }

def make_landing_burn_marks(scale=1.0, density=3000):
    """Create rocket engine burn marks on moon surface."""
    spacing = 1.0 / np.sqrt(density)
    points = []
    
    # Main engine burn mark (center)
    center_x, center_y = 0, 0
    burn_radius = 1.5 * scale
    
    for angle in np.arange(0, 2*np.pi, spacing/burn_radius):
        for r in np.arange(0, burn_radius, spacing):
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            z = -0.1 * (1 - r/burn_radius)  # Deeper in center
            points.append([x, y, z])
    
    # Side engine burn marks (4 smaller ones)
    side_positions = [
        (1.0*scale, 0, 0), (-1.0*scale, 0, 0),
        (0, 1.0*scale, 0), (0, -1.0*scale, 0)
    ]
    
    for sx, sy, sz in side_positions:
        side_radius = 0.6 * scale
        for angle in np.arange(0, 2*np.pi, spacing/side_radius):
            for r in np.arange(0, side_radius, spacing):
                x = sx + r * np.cos(angle)
                y = sy + r * np.sin(angle)
                z = -0.05 * (1 - r/side_radius)
                points.append([x, y, z])
    
    points = np.array(points)
    
    # Burn mark colors (dark/blackened)
    colors = np.full((len(points), 3), [60, 60, 70])
    
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 50),
        "cls": np.full(len(points), CLASS["ground"])
    }

def make_footprints_and_equipment(scale=1.0, density=3000):
    """Create footprints and small equipment on moon surface."""
    spacing = 1.0 / np.sqrt(density)
    points = []
    
    # Astronaut footprints leading from lander
    footprint_positions = [
        (2.5, 1.5, 0.01), (2.8, 1.8, 0.01), (3.1, 2.1, 0.01), (3.4, 2.4, 0.01),
        (3.7, 2.7, 0.01), (4.0, 3.0, 0.01), (4.3, 3.3, 0.01), (4.6, 3.6, 0.01)
    ]
    
    for fx, fy, fz in footprint_positions:
        # Create footprint impression
        for angle in np.arange(0, 2*np.pi, spacing/0.15):
            for r in np.arange(0, 0.15, spacing):
                x = fx + r * np.cos(angle)
                y = fy + r * np.sin(angle)
                z = fz - 0.02 * (1 - r/0.15)  # Slight depression
                points.append([x, y, z])
    
    # Small equipment items
    # Camera tripod marks
    for cx, cy in [(5.2, -4.2), (5.0, -4.0), (5.4, -4.4)]:
        for angle in np.arange(0, 2*np.pi, spacing/0.05):
            x = cx + 0.05 * np.cos(angle)
            y = cy + 0.05 * np.sin(angle)
            z = 0.01
            points.append([x, y, z])
    
    # Flag pole base
    for angle in np.arange(0, 2*np.pi, spacing/0.1):
        x = -4.0 + 0.1 * np.cos(angle)
        y = 3.0 + 0.1 * np.sin(angle)
        z = 0.01
        points.append([x, y, z])
    
    # Lander foot pad impressions
    lander_foot_positions = [
        (1.5, 0, 0.01), (-1.5, 0, 0.01), (0, 1.5, 0.01), (0, -1.5, 0.01)
    ]
    
    for lx, ly, lz in lander_foot_positions:
        for angle in np.arange(0, 2*np.pi, spacing/0.3):
            for r in np.arange(0, 0.3, spacing):
                x = lx + r * np.cos(angle)
                y = ly + r * np.sin(angle)
                z = lz - 0.05 * (1 - r/0.3)  # Deeper depression
                points.append([x, y, z])
    
    points = np.array(points)
    
    # Footprint colors (darker than moon surface)
    colors = np.full((len(points), 3), [120, 120, 130])
    
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 100),
        "cls": np.full(len(points), CLASS["ground"])
    }

def make_video_camera(scale=0.5, density=5000):
    """Create video camera on tripod."""
    spacing = 1.0 / np.sqrt(density)
    points = []
    
    # Tripod legs
    leg_positions = [
        (0.3*scale, 0, 0), (-0.15*scale, 0.26*scale, 0), (-0.15*scale, -0.26*scale, 0)
    ]
    
    for leg_x, leg_y, leg_z in leg_positions:
        for t in np.arange(0, 1, spacing):
            x = leg_x * (1 - t)
            y = leg_y * (1 - t)
            z = 0.8 * scale * t
            points.append([x, y, z])
    
    # Camera body
    camera_height = 0.3 * scale
    camera_width = 0.2 * scale
    camera_depth = 0.15 * scale
    
    for x in np.arange(-camera_width/2, camera_width/2, spacing):
        for y in np.arange(-camera_depth/2, camera_depth/2, spacing):
            for z in np.arange(0.8*scale, 0.8*scale + camera_height, spacing):
                points.append([x, y, z])
    
    # Lens
    lens_radius = 0.08 * scale
    lens_x = camera_width/2 + 0.05*scale
    for phi in np.arange(0, np.pi/2, spacing/lens_radius):
        for theta in np.arange(0, 2*np.pi, spacing/lens_radius):
            x = lens_x + lens_radius * np.sin(phi) * np.cos(theta)
            y = lens_radius * np.sin(phi) * np.sin(theta)
            z = 0.8*scale + camera_height/2 + lens_radius * np.cos(phi)
            points.append([x, y, z])
    
    points = np.array(points)
    
    # Camera color (black)
    colors = np.full((len(points), 3), [30, 30, 30])
    
    return {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "red": colors[:, 0],
        "green": colors[:, 1],
        "blue": colors[:, 2],
        "intensity": np.full(len(points), 100),
        "cls": np.full(len(points), CLASS["building"])
    }

def create_moon_landing_scene():
    """Create the complete moon landing scene."""
    print("🌙 Creating Moon Landing Scene...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate timestamp
    timestamp = int(datetime.now().timestamp())
    
    # Create moon surface
    print("  Creating moon surface...")
    moon_surface = make_moon_surface(MOON_SURFACE_SIZE, MOON_DENSITY)
    
    # Create lunar lander (positioned on moon surface)
    print("  Creating lunar lander...")
    lander = make_lunar_lander(LANDER_SCALE, OBJECT_DENSITY)
    lander = translate(lander, 0, 0, 1.0)  # Higher above surface to sit properly
    
    # Create astronaut (near lander)
    print("  Creating astronaut...")
    astronaut = make_astronaut(ASTRONAUT_SCALE, OBJECT_DENSITY)
    astronaut = translate(astronaut, 3, 2, 0)  # To the right of lander
    
    # Create American flag (planted in moon surface)
    print("  Creating American flag...")
    flag = make_american_flag(FLAG_SCALE, OBJECT_DENSITY)
    flag = translate(flag, -4, 3, 0)  # To the left of lander
    
    # Create video camera (recording the scene)
    print("  Creating video camera...")
    camera = make_video_camera(CAMERA_SCALE, OBJECT_DENSITY)
    camera = translate(camera, 5, -4, -.75)  # In front of the scene
    
    # Create footprints and surface details
    print("  Creating footprints and surface details...")
    footprints = make_footprints_and_equipment(1.0, 3000)
    
    # Create additional detailed elements
    print("  Creating equipment bag...")
    equipment_bag = make_equipment_bag(0.3, 4000)
    equipment_bag = translate(equipment_bag, 2, 1, 0)  # Near lander
    
    print("  Creating scattered rocks...")
    rocks = make_scattered_rocks(1.0, 2000)
    
    print("  Creating lander ladder...")
    ladder = make_lander_ladder(0.8, 5000)
    ladder = translate(ladder, 0, 0, 0.5)  # Same height as lander
    
    print("  Creating landing burn marks...")
    burn_marks = make_landing_burn_marks(1.0, 3000)
    
    # Combine all elements
    print("  Combining scene elements...")
    scene_data = stack_fields([moon_surface, lander, astronaut, flag, camera, footprints, 
                              equipment_bag, rocks, ladder, burn_marks])
    
    # Ensure classification keys
    scene_data = ensure_cls(scene_data)
    
    # Check point count
    total_points = len(scene_data["x"])
    print(f"  Total points generated: {total_points:,}")
    
    if total_points > MAX_POINTS:
        print(f"  WARNING: Point count ({total_points:,}) exceeds {MAX_POINTS:,}!")
        print("  Consider reducing density or scene size.")
        return
    
    # Create LAS file
    print("  Writing LAS file...")
    las_filename = f"{OUTPUT_DIR}/{SCENE_NAME}_{timestamp}.laz"
    
    # Create LAS header
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.x_scale = 0.01
    header.y_scale = 0.01
    header.z_scale = 0.01
    header.x_offset = 0
    header.y_offset = 0
    header.z_offset = 0
    
    # Create LAS file
    las_file = laspy.LasData(header)
    las_file.x = scene_data["x"]
    las_file.y = scene_data["y"]
    las_file.z = scene_data["z"]
    las_file.red = scene_data["red"]
    las_file.green = scene_data["green"]
    las_file.blue = scene_data["blue"]
    las_file.intensity = scene_data["intensity"]
    las_file.classification = scene_data["cls"]
    
    # Write file
    las_file.write(las_filename)
    print(f"  ✅ LAS file written: {las_filename}")
    
    # Create metadata
    metadata = {
        "scene_name": SCENE_NAME,
        "timestamp": timestamp,
        "total_points": total_points,
        "moon_surface_size": MOON_SURFACE_SIZE,
        "lander_scale": LANDER_SCALE,
        "astronaut_scale": ASTRONAUT_SCALE,
        "flag_scale": FLAG_SCALE,
        "camera_scale": CAMERA_SCALE,
        "moon_density": MOON_DENSITY,
        "object_density": OBJECT_DENSITY,
        "detail_density": DETAIL_DENSITY,
        "elements": [
            "Moon surface with realistic craters and varied colors",
            "Apollo-style lunar lander with antennas and thrusters",
            "Astronaut in spacesuit",
            "American flag on pole with rippled fabric",
            "Video camera on tripod",
            "Astronaut footprints and equipment marks",
            "Equipment bag/tool kit near lander",
            "Scattered rocks and debris around landing site",
            "Lander ladder from surface to cabin",
            "Rocket engine burn marks on moon surface"
        ],
        "description": "Point cloud representation of the iconic Apollo 11 moon landing scene"
    }
    
    metadata_filename = f"{OUTPUT_DIR}/{SCENE_NAME}_metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Metadata written: {metadata_filename}")
    
    # Create preview
    print("  Creating 3D preview...")
    try:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        points = np.column_stack([scene_data["x"], scene_data["y"], scene_data["z"]])
        colors = np.column_stack([scene_data["red"], scene_data["green"], scene_data["blue"]]) / 255.0
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Moon Landing Scene", width=1200, height=800)
        vis.add_geometry(pcd)
        
        # Set camera position for good view
        ctr = vis.get_view_control()
        ctr.set_front([0.5, -0.5, 0.7])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.3)
        
        # Run visualizer
        print("  🌙 Moon Landing Scene Preview:")
        print("    - Moon surface with realistic craters and varied colors")
        print("    - Apollo-style lunar lander with antennas and thrusters")
        print("    - Astronaut in white spacesuit")
        print("    - American flag planted in moon surface")
        print("    - Video camera recording the historic moment")
        print("    - Astronaut footprints and equipment marks on surface")
        print("    - Equipment bag and tools near the lander")
        print("    - Scattered rocks and debris around landing site")
        print("    - Lander ladder from surface to cabin")
        print("    - Rocket engine burn marks on moon surface")
        print("  Close the preview window to continue...")
        
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"  ⚠️  Preview failed: {e}")
        print("  LAS file created successfully, but preview unavailable.")
    
    # Create zip file
    zip_filename = f"{OUTPUT_DIR}/{SCENE_NAME}_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(las_filename, os.path.basename(las_filename))
        zipf.write(metadata_filename, os.path.basename(metadata_filename))
    
    print(f"  ✅ Zip file created: {zip_filename}")
    print(f"\n🌙 Moon Landing Scene Complete!")
    print(f"   Total Points: {total_points:,}")
    print(f"   Files: {las_filename}, {zip_filename}")
    
    return las_filename

if __name__ == "__main__":
    create_moon_landing_scene()
