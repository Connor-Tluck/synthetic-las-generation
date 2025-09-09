"""
Synthetic Scene Generation Module

This module contains all the individual scene generation functions for creating
realistic urban infrastructure elements. Each function generates a specific
type of infrastructure element that can be composed into larger scenes.
"""

import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

# Import the existing scene generation functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    make_storm_inlet_grate, MAT_INT, MAT_RGB_255, CLASS, RNG, BASE_SPACING,
    XY_JITTER, Z_JITTER
)


class SceneGenerator:
    """
    Professional scene generator for creating realistic urban infrastructure scenes.
    
    This class provides methods to generate individual infrastructure elements
    and compose them into realistic urban environments.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the scene generator with configuration.
        
        Args:
            config: Configuration dictionary with generation parameters
        """
        self.config = config or {}
        self.scene_library = self._build_scene_library()
        
    def _build_scene_library(self) -> Dict[str, callable]:
        """Build the library of available scene generation functions."""
        return {
            # Road infrastructure
            "street_patch": lambda **kwargs: make_street_patch(
                kwargs.get('width', 8.0), kwargs.get('length', 10.0), 
                kwargs.get('base_z', 0.0), kwargs.get('slope', (0.0, 0.01))
            ),
            "crosswalk": lambda **kwargs: make_crosswalk(
                kwargs.get('width', 6.0), kwargs.get('length', 10.0), 
                kwargs.get('base_z', 0.0)
            ),
            "curb_with_road": lambda **kwargs: make_curb(
                kwargs.get('run_len', 10.0), kwargs.get('curb_w', 0.3), 
                kwargs.get('curb_h', 0.15), kwargs.get('base_z', 0.0), 
                kwargs.get('road_slope', (0.0, 0.002))
            ),
            "sidewalk": lambda **kwargs: make_sidewalk(
                kwargs.get('width', 3.0), kwargs.get('length', 10.0), 
                kwargs.get('base_z', 0.0)
            ),
            "speed_hump": lambda **kwargs: make_speed_hump(
                kwargs.get('width', 6.0), kwargs.get('length', 8.0), 
                kwargs.get('height', 0.12), kwargs.get('base_z', 0.0)
            ),
            "driveway_crown": lambda **kwargs: make_driveway_crown(
                kwargs.get('width', 6.0), kwargs.get('length', 8.0), 
                kwargs.get('crown_h', 0.07), kwargs.get('base_z', 0.0)
            ),
            "street_banked": lambda **kwargs: make_street_patch(
                kwargs.get('width', 8.0), kwargs.get('length', 10.0), 
                kwargs.get('base_z', 0.0), kwargs.get('slope', (0.02, 0.0))
            ),
            
            # Barriers and safety
            "jersey_barrier": lambda **kwargs: make_barrier_jersey(
                kwargs.get('run_len', 10.0), kwargs.get('base_z', 0.0)
            ),
            "guardrail": lambda **kwargs: make_guardrail(
                kwargs.get('run_len', 10.0), kwargs.get('base_z', 0.0)
            ),
            "bollards": lambda **kwargs: make_bollards(
                kwargs.get('count', 6), kwargs.get('base_z', 0.0)
            ),
            
            # Utilities and infrastructure
            "power_pole_wires": lambda **kwargs: make_power_pole_and_lines(
                kwargs.get('base_z', 0.0)
            ),
            "streetlight": lambda **kwargs: make_streetlight(
                kwargs.get('base_z', 0.0)
            ),
            "utility_cabinet": lambda **kwargs: make_utility_box(
                kwargs.get('base_z', 0.0)
            ),
            "phone_cabinet": lambda **kwargs: make_phone_cabinet(
                kwargs.get('base_z', 0.0)
            ),
            "manholes": lambda **kwargs: make_manholes_and_grates(
                kwargs.get('count', 4), kwargs.get('base_z', 0.0)
            ),
            "storm_inlet_grate": lambda **kwargs: make_storm_inlet_grate(
                kwargs.get('base_z', 0.0)
            ),
            
            # Traffic control
            "stop_sign": lambda **kwargs: make_stop_sign(
                kwargs.get('base_z', 0.0)
            ),
            "parking_bumpers": lambda **kwargs: make_parking_bumpers(
                kwargs.get('count', 4), kwargs.get('base_z', 0.0)
            ),
            
            # Vegetation and landscaping
            "tree": lambda **kwargs: make_tree(
                kwargs.get('base_z', 0.0)
            ),
            
            # Street furniture
            "bench": lambda **kwargs: make_bench(
                kwargs.get('base_z', 0.0)
            ),
            "trash_can": lambda **kwargs: make_trash_can(
                kwargs.get('base_z', 0.0)
            ),
            "bike_rack_u": lambda **kwargs: make_bike_rack_u(
                kwargs.get('base_z', 0.0)
            ),
            "picnic_table": lambda **kwargs: make_picnic_table(
                kwargs.get('base_z', 0.0)
            ),
            
            # Emergency and services
            "fire_hydrant": lambda **kwargs: make_fire_hydrant(
                kwargs.get('base_z', 0.0)
            ),
            "mailbox_cluster": lambda **kwargs: make_mailbox_cluster(
                kwargs.get('base_z', 0.0)
            ),
            
            # Accessibility
            "stairs": lambda **kwargs: make_stairs(
                kwargs.get('width', 2.0), kwargs.get('depth', 0.3), 
                kwargs.get('step_h', 0.15), kwargs.get('n_steps', 12), 
                kwargs.get('base_z', 0.0), "concrete", "building"
            ),
            "ped_ramp_tactile": lambda **kwargs: make_ped_ramp_with_tactile(
                kwargs.get('width', 1.8), kwargs.get('length', 2.0), 
                kwargs.get('rise', 0.15), kwargs.get('base_z', 0.0)
            ),
            
            # Complex features
            "median_island": lambda **kwargs: make_median_island_with_curbs(
                kwargs.get('length', 4.0), kwargs.get('width', 1.6), 
                kwargs.get('curb_h', 0.12), kwargs.get('base_z', 0.0)
            ),
            "furniture_boxes": lambda **kwargs: stack_fields([
                translate(make_box(1.0, 0.6, 0.7, 0.0, "wood", "unclassified"), 0.0, 0.0, 0.0),
                translate(make_box(0.8, 0.8, 1.0, 0.0, "plastic", "unclassified"), 1.4, 0.2, 0.0)
            ])
        }
    
    def generate_scene(self, scene_type: str, **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate a specific scene type with given parameters.
        
        Args:
            scene_type: Type of scene to generate
            **kwargs: Parameters for scene generation
            
        Returns:
            Dictionary containing point cloud data
        """
        if scene_type not in self.scene_library:
            raise ValueError(f"Unknown scene type: {scene_type}")
        
        scene_data = self.scene_library[scene_type](**kwargs)
        return ensure_cls(scene_data)
    
    def get_available_scenes(self) -> List[str]:
        """Get list of available scene types."""
        return list(self.scene_library.keys())
    
    def get_scene_info(self, scene_type: str) -> Dict[str, Any]:
        """
        Get information about a scene type including default parameters.
        
        Args:
            scene_type: Type of scene to get info for
            
        Returns:
            Dictionary with scene information
        """
        if scene_type not in self.scene_library:
            raise ValueError(f"Unknown scene type: {scene_type}")
        
        # Default parameters for each scene type
        defaults = {
            "street_patch": {"width": 8.0, "length": 10.0, "base_z": 0.0, "slope": (0.0, 0.01)},
            "crosswalk": {"width": 6.0, "length": 10.0, "base_z": 0.0},
            "curb_with_road": {"run_len": 10.0, "curb_w": 0.3, "curb_h": 0.15, "base_z": 0.0, "road_slope": (0.0, 0.002)},
            "sidewalk": {"width": 3.0, "length": 10.0, "base_z": 0.0},
            "speed_hump": {"width": 6.0, "length": 8.0, "height": 0.12, "base_z": 0.0},
            "driveway_crown": {"width": 6.0, "length": 8.0, "crown_h": 0.07, "base_z": 0.0},
            "street_banked": {"width": 8.0, "length": 10.0, "base_z": 0.0, "slope": (0.02, 0.0)},
            "jersey_barrier": {"run_len": 10.0, "base_z": 0.0},
            "guardrail": {"run_len": 10.0, "base_z": 0.0},
            "bollards": {"count": 6, "base_z": 0.0},
            "power_pole_wires": {"base_z": 0.0},
            "streetlight": {"base_z": 0.0},
            "utility_cabinet": {"base_z": 0.0},
            "phone_cabinet": {"base_z": 0.0},
            "manholes": {"count": 4, "base_z": 0.0},
            "storm_inlet_grate": {"base_z": 0.0},
            "stop_sign": {"base_z": 0.0},
            "parking_bumpers": {"count": 4, "base_z": 0.0},
            "tree": {"base_z": 0.0},
            "bench": {"base_z": 0.0},
            "trash_can": {"base_z": 0.0},
            "bike_rack_u": {"base_z": 0.0},
            "picnic_table": {"base_z": 0.0},
            "fire_hydrant": {"base_z": 0.0},
            "mailbox_cluster": {"base_z": 0.0},
            "stairs": {"width": 2.0, "depth": 0.3, "step_h": 0.15, "n_steps": 12, "base_z": 0.0},
            "ped_ramp_tactile": {"width": 1.8, "length": 2.0, "rise": 0.15, "base_z": 0.0},
            "median_island": {"length": 4.0, "width": 1.6, "curb_h": 0.12, "base_z": 0.0},
            "furniture_boxes": {"base_z": 0.0}
        }
        
        return {
            "type": scene_type,
            "defaults": defaults.get(scene_type, {}),
            "description": self._get_scene_description(scene_type)
        }
    
    def _get_scene_description(self, scene_type: str) -> str:
        """Get human-readable description of scene type."""
        descriptions = {
            "street_patch": "Basic asphalt road surface",
            "crosswalk": "Road with white painted crosswalk markings",
            "curb_with_road": "Road section with concrete curb",
            "sidewalk": "Concrete pedestrian walkway",
            "speed_hump": "Raised road surface for traffic calming",
            "driveway_crown": "Curved road surface with crown",
            "street_banked": "Angled road surface for drainage",
            "jersey_barrier": "Concrete traffic barrier",
            "guardrail": "Metal safety barrier along roads",
            "bollards": "Vertical traffic posts for protection",
            "power_pole_wires": "Utility pole with sagging power lines",
            "streetlight": "Road lighting fixture",
            "utility_cabinet": "Metal utility equipment box",
            "phone_cabinet": "Telecommunications equipment",
            "manholes": "Metal utility access covers",
            "storm_inlet_grate": "Drainage system access",
            "stop_sign": "Traffic control sign",
            "parking_bumpers": "Concrete parking space markers",
            "tree": "Vegetation with trunk and crown",
            "bench": "Wooden park bench",
            "trash_can": "Waste receptacle",
            "bike_rack_u": "Bicycle parking fixture",
            "picnic_table": "Outdoor seating",
            "fire_hydrant": "Emergency water access",
            "mailbox_cluster": "Postal collection point",
            "stairs": "Concrete staircase",
            "ped_ramp_tactile": "ADA-compliant ramp with tactile markings",
            "median_island": "Road divider with curbs",
            "furniture_boxes": "Various storage containers"
        }
        return descriptions.get(scene_type, "Unknown scene type")
