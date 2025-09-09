"""
Scene Composition Engine

This module handles the realistic composition of individual scene elements into
coherent urban environments following real-world placement rules and constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import math
from scene_config import SceneConfig, FeatureConfig, SceneConfigurationManager
from synthetic_scenes import SceneGenerator


@dataclass
class PlacementRule:
    """Defines placement rules for scene elements."""
    position: Tuple[float, float, float]
    rotation: float = 0.0
    scale: float = 1.0
    constraints: Dict[str, Any] = None


class SceneComposer:
    """
    Composes individual scene elements into realistic urban environments.
    
    Handles spatial relationships, collision detection, and realistic placement
    of infrastructure elements according to urban planning principles.
    """
    
    def __init__(self, scene_generator: SceneGenerator):
        """
        Initialize the scene composer.
        
        Args:
            scene_generator: SceneGenerator instance for creating individual elements
        """
        self.scene_generator = scene_generator
        self.placement_history = []
        self.collision_buffer = 0.5  # meters
        
    def compose_scene(self, config: SceneConfig) -> Dict[str, np.ndarray]:
        """
        Compose a complete scene from configuration.
        
        Args:
            config: Scene configuration defining the scene to create
            
        Returns:
            Dictionary containing the complete composed point cloud data
        """
        self.placement_history = []
        scene_parts = []
        
        # Sort features by priority (roads first, then infrastructure, then details)
        feature_priority = self._get_feature_priority()
        sorted_features = sorted(
            config.features.items(),
            key=lambda x: feature_priority.get(x[0], 999)
        )
        
        for feature_name, feature_config in sorted_features:
            if not feature_config.enabled:
                continue
                
            for i in range(feature_config.count):
                placement = self._calculate_placement(
                    feature_name, feature_config, config, i
                )
                
                if placement:
                    scene_data = self._generate_and_place_feature(
                        feature_name, feature_config, placement
                    )
                    if scene_data:
                        scene_parts.append(scene_data)
                        self.placement_history.append({
                            'feature': feature_name,
                            'placement': placement,
                            'bounds': self._get_feature_bounds(scene_data)
                        })
        
        # Combine all scene parts
        if scene_parts:
            return self._combine_scene_parts(scene_parts)
        else:
            return self._create_empty_scene()
    
    def _get_feature_priority(self) -> Dict[str, int]:
        """Get placement priority for different feature types."""
        return {
            # Infrastructure (highest priority)
            'street_patch': 1,
            'crosswalk': 2,
            'curb_with_road': 3,
            'sidewalk': 4,
            
            # Major utilities
            'power_pole_wires': 10,
            'streetlight': 11,
            'utility_cabinet': 12,
            'phone_cabinet': 13,
            
            # Safety and barriers
            'jersey_barrier': 20,
            'guardrail': 21,
            'bollards': 22,
            
            # Traffic control
            'stop_sign': 30,
            'speed_hump': 31,
            'parking_bumpers': 32,
            
            # Vegetation
            'tree': 40,
            
            # Street furniture
            'bench': 50,
            'trash_can': 51,
            'bike_rack_u': 52,
            'picnic_table': 53,
            
            # Utilities and services
            'fire_hydrant': 60,
            'mailbox_cluster': 61,
            'manholes': 62,
            'storm_inlet_grate': 63,
            
            # Accessibility
            'stairs': 70,
            'ped_ramp_tactile': 71,
            
            # Complex features
            'median_island': 80,
            'furniture_boxes': 81,
            'driveway_crown': 82,
            'street_banked': 83
        }
    
    def _calculate_placement(self, feature_name: str, feature_config: FeatureConfig,
                           config: SceneConfig, instance_index: int) -> Optional[PlacementRule]:
        """
        Calculate realistic placement for a feature instance.
        
        Args:
            feature_name: Name of the feature type
            feature_config: Configuration for this feature
            config: Overall scene configuration
            instance_index: Index of this instance (0-based)
            
        Returns:
            PlacementRule with position and constraints, or None if placement failed
        """
        placement_rules = feature_config.placement_rules
        
        if feature_name == 'street_patch':
            return self._place_road(feature_config, config, instance_index)
        elif feature_name == 'crosswalk':
            return self._place_crosswalk(feature_config, config, instance_index)
        elif feature_name == 'sidewalk':
            return self._place_sidewalk(feature_config, config, instance_index)
        elif feature_name == 'curb_with_road':
            return self._place_curb(feature_config, config, instance_index)
        elif feature_name == 'tree':
            return self._place_tree(feature_config, config, instance_index)
        elif feature_name == 'streetlight':
            return self._place_streetlight(feature_config, config, instance_index)
        elif feature_name == 'utility_cabinet':
            return self._place_utility_cabinet(feature_config, config, instance_index)
        elif feature_name == 'fire_hydrant':
            return self._place_fire_hydrant(feature_config, config, instance_index)
        elif feature_name == 'mailbox_cluster':
            return self._place_mailbox_cluster(feature_config, config, instance_index)
        elif feature_name == 'power_pole_wires':
            return self._place_power_pole(feature_config, config, instance_index)
        elif feature_name == 'stop_sign':
            return self._place_stop_sign(feature_config, config, instance_index)
        elif feature_name == 'manholes':
            return self._place_manholes(feature_config, config, instance_index)
        elif feature_name == 'bollards':
            return self._place_bollards(feature_config, config, instance_index)
        elif feature_name == 'jersey_barrier':
            return self._place_jersey_barrier(feature_config, config, instance_index)
        elif feature_name == 'guardrail':
            return self._place_guardrail(feature_config, config, instance_index)
        elif feature_name == 'bench':
            return self._place_bench(feature_config, config, instance_index)
        elif feature_name == 'trash_can':
            return self._place_trash_can(feature_config, config, instance_index)
        elif feature_name == 'bike_rack_u':
            return self._place_bike_rack(feature_config, config, instance_index)
        elif feature_name == 'picnic_table':
            return self._place_picnic_table(feature_config, config, instance_index)
        else:
            return self._place_generic(feature_config, config, instance_index)
    
    def _place_road(self, feature_config: FeatureConfig, config: SceneConfig, 
                   instance_index: int) -> PlacementRule:
        """Place road elements at scene center."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        if 'center_x' in feature_config.placement_rules:
            x = scene_width / 2
        else:
            x = scene_width / 2  # Default to center
        
        if 'center_y' in feature_config.placement_rules:
            y = scene_length / 2
        else:
            y = scene_length / 2  # Default to center
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_sidewalk(self, feature_config: FeatureConfig, config: SceneConfig,
                       instance_index: int) -> PlacementRule:
        """Place sidewalks alongside roads."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        offset = feature_config.placement_rules.get('offset_from_road', 5.0)
        side = feature_config.placement_rules.get('side', 'both')
        
        if side == 'both':
            # Alternate sides for multiple sidewalks
            if instance_index % 2 == 0:
                x = scene_width / 2 - offset
            else:
                x = scene_width / 2 + offset
        elif side == 'left':
            x = scene_width / 2 - offset
        elif side == 'right':
            x = scene_width / 2 + offset
        else:
            x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
        
        return PlacementRule(position=(x, scene_length / 2, 0.0))
    
    def _place_curb(self, feature_config: FeatureConfig, config: SceneConfig,
                   instance_index: int) -> PlacementRule:
        """Place curbs between roads and sidewalks."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        offset = feature_config.placement_rules.get('offset_from_road', 4.0)
        side = feature_config.placement_rules.get('side', 'both')
        
        if side == 'both':
            if instance_index % 2 == 0:
                x = scene_width / 2 - offset
            else:
                x = scene_width / 2 + offset
        elif side == 'left':
            x = scene_width / 2 - offset
        elif side == 'right':
            x = scene_width / 2 + offset
        else:
            x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
        
        return PlacementRule(position=(x, scene_length / 2, 0.0))
    
    def _place_tree(self, feature_config: FeatureConfig, config: SceneConfig,
                   instance_index: int) -> PlacementRule:
        """Place trees with natural spacing and clustering."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        spacing = feature_config.placement_rules.get('spacing', 8.0)
        offset = feature_config.placement_rules.get('offset_from_sidewalk', 1.0)
        side = feature_config.placement_rules.get('side', 'both')
        
        # Calculate grid positions
        trees_per_side = feature_config.count // 2 if side == 'both' else feature_config.count
        trees_per_row = int(scene_length / spacing)
        
        if trees_per_row == 0:
            trees_per_row = 1
        
        row = instance_index // trees_per_row
        col = instance_index % trees_per_row
        
        y = (col + 1) * (scene_length / (trees_per_row + 1))
        
        if side == 'both':
            if instance_index % 2 == 0:
                x = scene_width / 2 - offset
            else:
                x = scene_width / 2 + offset
        elif side == 'left':
            x = scene_width / 2 - offset
        elif side == 'right':
            x = scene_width / 2 + offset
        else:
            x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
        
        # Add some natural variation
        x += np.random.normal(0, 0.5)
        y += np.random.normal(0, 0.5)
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_streetlight(self, feature_config: FeatureConfig, config: SceneConfig,
                          instance_index: int) -> PlacementRule:
        """Place streetlights with regular spacing."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        spacing = feature_config.placement_rules.get('spacing', 20.0)
        offset = feature_config.placement_rules.get('offset_from_road', 3.0)
        side = feature_config.placement_rules.get('side', 'alternating')
        
        lights_per_side = feature_config.count // 2 if side in ['both', 'alternating'] else feature_config.count
        lights_per_row = int(scene_length / spacing)
        
        if lights_per_row == 0:
            lights_per_row = 1
        
        row = instance_index // lights_per_row
        col = instance_index % lights_per_row
        
        y = (col + 1) * (scene_length / (lights_per_row + 1))
        
        if side == 'both':
            if instance_index % 2 == 0:
                x = scene_width / 2 - offset
            else:
                x = scene_width / 2 + offset
        elif side == 'alternating':
            if instance_index % 2 == 0:
                x = scene_width / 2 - offset
            else:
                x = scene_width / 2 + offset
        elif side == 'left':
            x = scene_width / 2 - offset
        elif side == 'right':
            x = scene_width / 2 + offset
        else:
            x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_utility_cabinet(self, feature_config: FeatureConfig, config: SceneConfig,
                              instance_index: int) -> PlacementRule:
        """Place utility cabinets with appropriate spacing."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        offset = feature_config.placement_rules.get('offset_from_sidewalk', 2.0)
        side = feature_config.placement_rules.get('side', 'random')
        
        # Place cabinets with good spacing
        spacing = max(scene_length / feature_config.count, 15.0)
        y = (instance_index + 1) * spacing
        
        if side == 'random':
            x = scene_width / 2 + (np.random.choice([-1, 1]) * offset)
        elif side == 'left':
            x = scene_width / 2 - offset
        elif side == 'right':
            x = scene_width / 2 + offset
        else:
            x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_fire_hydrant(self, feature_config: FeatureConfig, config: SceneConfig,
                           instance_index: int) -> PlacementRule:
        """Place fire hydrants with appropriate spacing."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        offset = feature_config.placement_rules.get('offset_from_curb', 1.0)
        side = feature_config.placement_rules.get('side', 'random')
        
        spacing = max(scene_length / feature_config.count, 25.0)
        y = (instance_index + 1) * spacing
        
        if side == 'random':
            x = scene_width / 2 + (np.random.choice([-1, 1]) * offset)
        elif side == 'left':
            x = scene_width / 2 - offset
        elif side == 'right':
            x = scene_width / 2 + offset
        else:
            x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_mailbox_cluster(self, feature_config: FeatureConfig, config: SceneConfig,
                              instance_index: int) -> PlacementRule:
        """Place mailbox clusters near sidewalks."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        offset = feature_config.placement_rules.get('offset_from_sidewalk', 1.0)
        side = feature_config.placement_rules.get('side', 'both')
        
        spacing = max(scene_length / feature_config.count, 20.0)
        y = (instance_index + 1) * spacing
        
        if side == 'both':
            if instance_index % 2 == 0:
                x = scene_width / 2 - offset
            else:
                x = scene_width / 2 + offset
        elif side == 'left':
            x = scene_width / 2 - offset
        elif side == 'right':
            x = scene_width / 2 + offset
        else:
            x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_power_pole(self, feature_config: FeatureConfig, config: SceneConfig,
                         instance_index: int) -> PlacementRule:
        """Place power poles with appropriate spacing."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        spacing = feature_config.placement_rules.get('spacing', 25.0)
        offset = feature_config.placement_rules.get('offset_from_road', 4.0)
        side = feature_config.placement_rules.get('side', 'alternating')
        
        poles_per_side = feature_config.count // 2 if side == 'alternating' else feature_config.count
        poles_per_row = int(scene_length / spacing)
        
        if poles_per_row == 0:
            poles_per_row = 1
        
        row = instance_index // poles_per_row
        col = instance_index % poles_per_row
        
        y = (col + 1) * (scene_length / (poles_per_row + 1))
        
        if side == 'alternating':
            if instance_index % 2 == 0:
                x = scene_width / 2 - offset
            else:
                x = scene_width / 2 + offset
        elif side == 'left':
            x = scene_width / 2 - offset
        elif side == 'right':
            x = scene_width / 2 + offset
        else:
            x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_stop_sign(self, feature_config: FeatureConfig, config: SceneConfig,
                        instance_index: int) -> PlacementRule:
        """Place stop signs at intersections or corners."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        offset = feature_config.placement_rules.get('offset_from_road', 2.0)
        
        # Place at corners for intersections
        if 'intersection_corners' in feature_config.placement_rules:
            corners = [
                (offset, offset),  # Bottom-left
                (scene_width - offset, offset),  # Bottom-right
                (offset, scene_length - offset),  # Top-left
                (scene_width - offset, scene_length - offset)  # Top-right
            ]
            if instance_index < len(corners):
                x, y = corners[instance_index]
                return PlacementRule(position=(x, y, 0.0))
        
        # Default placement along road
        spacing = max(scene_length / feature_config.count, 20.0)
        y = (instance_index + 1) * spacing
        x = scene_width / 2 + offset
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_manholes(self, feature_config: FeatureConfig, config: SceneConfig,
                       instance_index: int) -> PlacementRule:
        """Place manholes in road surface."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        if 'road_surface' in feature_config.placement_rules:
            spacing = feature_config.placement_rules.get('spacing', 15.0)
            y = (instance_index + 1) * spacing
            x = scene_width / 2 + np.random.normal(0, 1.0)  # Random lateral position
            return PlacementRule(position=(x, y, 0.0))
        
        return self._place_generic(feature_config, config, instance_index)
    
    def _place_bollards(self, feature_config: FeatureConfig, config: SceneConfig,
                       instance_index: int) -> PlacementRule:
        """Place bollards for protection."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        offset = feature_config.placement_rules.get('offset_from_curb', 1.0)
        side = feature_config.placement_rules.get('side', 'all')
        
        spacing = max(scene_length / feature_config.count, 8.0)
        y = (instance_index + 1) * spacing
        
        if side == 'all':
            # Distribute around perimeter
            if instance_index < feature_config.count // 4:
                x = scene_width / 2 - offset  # Left side
            elif instance_index < feature_config.count // 2:
                x = scene_width / 2 + offset  # Right side
            elif instance_index < 3 * feature_config.count // 4:
                x = scene_width / 2 - offset  # Left side (second row)
            else:
                x = scene_width / 2 + offset  # Right side (second row)
        else:
            x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_jersey_barrier(self, feature_config: FeatureConfig, config: SceneConfig,
                             instance_index: int) -> PlacementRule:
        """Place jersey barriers in median or along roads."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        if 'center_median' in feature_config.placement_rules:
            x = scene_width / 2
            y = scene_length / 2
            return PlacementRule(position=(x, y, 0.0))
        
        return self._place_generic(feature_config, config, instance_index)
    
    def _place_guardrail(self, feature_config: FeatureConfig, config: SceneConfig,
                        instance_index: int) -> PlacementRule:
        """Place guardrails along shoulders."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        if 'shoulder' in feature_config.placement_rules:
            side = feature_config.placement_rules.get('side', 'both')
            offset = 8.0  # Shoulder offset
            
            if side == 'both':
                if instance_index % 2 == 0:
                    x = scene_width / 2 - offset
                else:
                    x = scene_width / 2 + offset
            elif side == 'left':
                x = scene_width / 2 - offset
            elif side == 'right':
                x = scene_width / 2 + offset
            else:
                x = scene_width / 2 + offset if instance_index % 2 == 0 else scene_width / 2 - offset
            
            y = scene_length / 2
            return PlacementRule(position=(x, y, 0.0))
        
        return self._place_generic(feature_config, config, instance_index)
    
    def _place_bench(self, feature_config: FeatureConfig, config: SceneConfig,
                    instance_index: int) -> PlacementRule:
        """Place benches near paths or sidewalks."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        spacing = feature_config.placement_rules.get('spacing', 15.0)
        y = (instance_index + 1) * spacing
        
        # Place near sidewalks
        x = scene_width / 2 + (np.random.choice([-1, 1]) * 3.0)
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_trash_can(self, feature_config: FeatureConfig, config: SceneConfig,
                        instance_index: int) -> PlacementRule:
        """Place trash cans near benches or paths."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        spacing = feature_config.placement_rules.get('spacing', 25.0)
        y = (instance_index + 1) * spacing
        
        # Place near sidewalks
        x = scene_width / 2 + (np.random.choice([-1, 1]) * 3.0)
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _place_bike_rack(self, feature_config: FeatureConfig, config: SceneConfig,
                        instance_index: int) -> PlacementRule:
        """Place bike racks near entrances or high-traffic areas."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        if 'near_entrances' in feature_config.placement_rules:
            # Place near scene corners (entrances)
            corners = [
                (scene_width * 0.2, scene_length * 0.2),
                (scene_width * 0.8, scene_length * 0.2),
                (scene_width * 0.2, scene_length * 0.8),
                (scene_width * 0.8, scene_length * 0.8)
            ]
            if instance_index < len(corners):
                x, y = corners[instance_index]
                return PlacementRule(position=(x, y, 0.0))
        
        return self._place_generic(feature_config, config, instance_index)
    
    def _place_picnic_table(self, feature_config: FeatureConfig, config: SceneConfig,
                           instance_index: int) -> PlacementRule:
        """Place picnic tables in groups."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        if 'grouped' in feature_config.placement_rules:
            spacing = feature_config.placement_rules.get('spacing', 20.0)
            # Create groups of 2-3 tables
            group_size = 3
            group_index = instance_index // group_size
            table_in_group = instance_index % group_size
            
            y = (group_index + 1) * spacing
            x = scene_width / 2 + (table_in_group - 1) * 5.0  # Spread within group
            
            return PlacementRule(position=(x, y, 0.0))
        
        return self._place_generic(feature_config, config, instance_index)
    
    def _place_crosswalk(self, feature_config: FeatureConfig, config: SceneConfig,
                        instance_index: int) -> PlacementRule:
        """Place crosswalks at intersections."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        if 'intersection_corners' in feature_config.placement_rules:
            # Place at intersection center
            x = scene_width / 2
            y = scene_length / 2
            return PlacementRule(position=(x, y, 0.0))
        
        return self._place_generic(feature_config, config, instance_index)
    
    def _place_generic(self, feature_config: FeatureConfig, config: SceneConfig,
                      instance_index: int) -> PlacementRule:
        """Generic placement for features without specific rules."""
        scene_width = config.scene_size['width']
        scene_length = config.scene_size['length']
        
        # Distribute evenly across the scene
        spacing = max(scene_length / feature_config.count, 5.0)
        y = (instance_index + 1) * spacing
        x = scene_width / 2 + np.random.normal(0, 2.0)
        
        return PlacementRule(position=(x, y, 0.0))
    
    def _generate_and_place_feature(self, feature_name: str, feature_config: FeatureConfig,
                                   placement: PlacementRule) -> Optional[Dict[str, np.ndarray]]:
        """
        Generate a feature and place it at the specified location.
        
        Args:
            feature_name: Name of the feature to generate
            feature_config: Configuration for the feature
            placement: Placement rule with position and constraints
            
        Returns:
            Point cloud data for the placed feature, or None if generation failed
        """
        try:
            # Generate the feature with configured parameters
            scene_data = self.scene_generator.generate_scene(
                feature_name, **feature_config.parameters
            )
            
            # Apply placement transformation
            if scene_data:
                scene_data = self._apply_placement(scene_data, placement)
                return scene_data
            
        except Exception as e:
            print(f"Warning: Failed to generate {feature_name}: {e}")
            return None
        
        return None
    
    def _apply_placement(self, scene_data: Dict[str, np.ndarray], 
                        placement: PlacementRule) -> Dict[str, np.ndarray]:
        """
        Apply placement transformation to scene data.
        
        Args:
            scene_data: Point cloud data
            placement: Placement rule with transformations
            
        Returns:
            Transformed point cloud data
        """
        if not scene_data:
            return scene_data
        
        # Apply translation
        dx, dy, dz = placement.position
        scene_data['x'] = scene_data['x'] + dx
        scene_data['y'] = scene_data['y'] + dy
        scene_data['z'] = scene_data['z'] + dz
        
        # Apply rotation (if needed)
        if placement.rotation != 0.0:
            # Simple 2D rotation around Z-axis
            cos_r = math.cos(placement.rotation)
            sin_r = math.sin(placement.rotation)
            
            x_new = scene_data['x'] * cos_r - scene_data['y'] * sin_r
            y_new = scene_data['x'] * sin_r + scene_data['y'] * cos_r
            
            scene_data['x'] = x_new
            scene_data['y'] = y_new
        
        # Apply scaling (if needed)
        if placement.scale != 1.0:
            scene_data['x'] = scene_data['x'] * placement.scale
            scene_data['y'] = scene_data['y'] * placement.scale
            scene_data['z'] = scene_data['z'] * placement.scale
        
        return scene_data
    
    def _get_feature_bounds(self, scene_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Get bounding box for a feature."""
        if not scene_data:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0, 'min_z': 0, 'max_z': 0}
        
        return {
            'min_x': float(np.min(scene_data['x'])),
            'max_x': float(np.max(scene_data['x'])),
            'min_y': float(np.min(scene_data['y'])),
            'max_y': float(np.max(scene_data['y'])),
            'min_z': float(np.min(scene_data['z'])),
            'max_z': float(np.max(scene_data['z']))
        }
    
    def _combine_scene_parts(self, scene_parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine multiple scene parts into a single point cloud."""
        if not scene_parts:
            return self._create_empty_scene()
        
        # Use the existing stack_fields function from the original code
        from generate_point_cloud_sandbox import stack_fields
        return stack_fields(scene_parts)
    
    def _create_empty_scene(self) -> Dict[str, np.ndarray]:
        """Create an empty scene with minimal structure."""
        return {
            'x': np.array([0.0]),
            'y': np.array([0.0]),
            'z': np.array([0.0]),
            'intensity': np.array([1000], dtype=np.uint16),
            'cls': np.array([1], dtype=np.uint8)
        }
