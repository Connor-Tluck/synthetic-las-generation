"""
Scene Configuration System

This module provides configuration management for large-scale synthetic scene generation.
It includes predefined realistic urban configurations and allows custom scene composition.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class SceneComplexity(Enum):
    """Scene complexity levels for different use cases."""
    MINIMAL = "minimal"
    BASIC = "basic"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class FeatureConfig:
    """Configuration for a specific feature type."""
    count: int
    parameters: Dict[str, Any]
    placement_rules: Dict[str, Any]
    enabled: bool = True


@dataclass
class SceneConfig:
    """Complete scene configuration."""
    name: str
    description: str
    complexity: SceneComplexity
    scene_size: Dict[str, float]  # width, length, height
    features: Dict[str, FeatureConfig]
    composition_rules: Dict[str, Any]
    output_settings: Dict[str, Any]


class SceneConfigurationManager:
    """
    Manages scene configurations for large-scale synthetic data generation.
    
    Provides predefined realistic urban configurations and allows custom scene composition.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to custom configuration file
        """
        self.config_file = config_file
        self.predefined_configs = self._load_predefined_configs()
        
    def _load_predefined_configs(self) -> Dict[str, SceneConfig]:
        """Load predefined realistic scene configurations."""
        configs = {}
        
        # Residential Street Configuration
        configs["residential_street"] = SceneConfig(
            name="Residential Street",
            description="Typical residential street with sidewalks, trees, and utilities",
            complexity=SceneComplexity.STANDARD,
            scene_size={"width": 50.0, "length": 100.0, "height": 15.0},
            features={
                "street_patch": FeatureConfig(
                    count=1,
                    parameters={"width": 8.0, "length": 100.0, "slope": (0.0, 0.01)},
                    placement_rules={"center_x": True, "center_y": True}
                ),
                "sidewalk": FeatureConfig(
                    count=2,
                    parameters={"width": 2.0, "length": 100.0},
                    placement_rules={"offset_from_road": 5.0, "side": "both"}
                ),
                "curb_with_road": FeatureConfig(
                    count=2,
                    parameters={"run_len": 100.0, "curb_w": 0.3, "curb_h": 0.15},
                    placement_rules={"offset_from_road": 4.0, "side": "both"}
                ),
                "tree": FeatureConfig(
                    count=20,
                    parameters={},
                    placement_rules={"spacing": 8.0, "offset_from_sidewalk": 1.0, "side": "both"}
                ),
                "streetlight": FeatureConfig(
                    count=10,
                    parameters={},
                    placement_rules={"spacing": 20.0, "offset_from_road": 3.0, "side": "alternating"}
                ),
                "utility_cabinet": FeatureConfig(
                    count=2,
                    parameters={},
                    placement_rules={"offset_from_sidewalk": 2.0, "side": "random"}
                ),
                "fire_hydrant": FeatureConfig(
                    count=3,
                    parameters={},
                    placement_rules={"offset_from_curb": 1.0, "side": "random"}
                ),
                "mailbox_cluster": FeatureConfig(
                    count=5,
                    parameters={},
                    placement_rules={"offset_from_sidewalk": 1.0, "side": "both"}
                )
            },
            composition_rules={
                "road_center": True,
                "sidewalk_symmetry": True,
                "utility_spacing": 20.0,
                "vegetation_density": 0.8
            },
            output_settings={
                "format": "laz",
                "compression": True,
                "include_metadata": True
            }
        )
        
        # Commercial Intersection Configuration
        configs["commercial_intersection"] = SceneConfig(
            name="Commercial Intersection",
            description="Busy commercial intersection with crosswalks, traffic control, and utilities",
            complexity=SceneComplexity.DETAILED,
            scene_size={"width": 80.0, "length": 80.0, "height": 20.0},
            features={
                "street_patch": FeatureConfig(
                    count=2,
                    parameters={"width": 12.0, "length": 80.0, "slope": (0.0, 0.01)},
                    placement_rules={"intersection": True, "perpendicular": True}
                ),
                "crosswalk": FeatureConfig(
                    count=4,
                    parameters={"width": 6.0, "length": 12.0},
                    placement_rules={"intersection_corners": True}
                ),
                "sidewalk": FeatureConfig(
                    count=4,
                    parameters={"width": 3.0, "length": 80.0},
                    placement_rules={"offset_from_road": 6.0, "side": "all"}
                ),
                "curb_with_road": FeatureConfig(
                    count=4,
                    parameters={"run_len": 80.0, "curb_w": 0.3, "curb_h": 0.15},
                    placement_rules={"offset_from_road": 6.0, "side": "all"}
                ),
                "stop_sign": FeatureConfig(
                    count=4,
                    parameters={},
                    placement_rules={"intersection_corners": True, "offset_from_road": 2.0}
                ),
                "streetlight": FeatureConfig(
                    count=8,
                    parameters={},
                    placement_rules={"spacing": 20.0, "offset_from_road": 3.0, "side": "all"}
                ),
                "power_pole_wires": FeatureConfig(
                    count=6,
                    parameters={},
                    placement_rules={"spacing": 25.0, "offset_from_road": 4.0, "side": "alternating"}
                ),
                "utility_cabinet": FeatureConfig(
                    count=3,
                    parameters={},
                    placement_rules={"offset_from_sidewalk": 2.0, "side": "random"}
                ),
                "manholes": FeatureConfig(
                    count=8,
                    parameters={"count": 1},
                    placement_rules={"road_surface": True, "spacing": 15.0}
                ),
                "bollards": FeatureConfig(
                    count=12,
                    parameters={"count": 3},
                    placement_rules={"offset_from_curb": 1.0, "side": "all"}
                )
            },
            composition_rules={
                "intersection_center": True,
                "traffic_control": True,
                "utility_underground": True,
                "pedestrian_safety": True
            },
            output_settings={
                "format": "laz",
                "compression": True,
                "include_metadata": True
            }
        )
        
        # Highway Configuration
        configs["highway_section"] = SceneConfig(
            name="Highway Section",
            description="Multi-lane highway with barriers, signage, and utilities",
            complexity=SceneComplexity.COMPREHENSIVE,
            scene_size={"width": 60.0, "length": 200.0, "height": 25.0},
            features={
                "street_patch": FeatureConfig(
                    count=4,
                    parameters={"width": 12.0, "length": 200.0, "slope": (0.0, 0.02)},
                    placement_rules={"lanes": 4, "spacing": 3.5}
                ),
                "jersey_barrier": FeatureConfig(
                    count=2,
                    parameters={"run_len": 200.0},
                    placement_rules={"center_median": True}
                ),
                "guardrail": FeatureConfig(
                    count=2,
                    parameters={"run_len": 200.0},
                    placement_rules={"shoulder": True, "side": "both"}
                ),
                "streetlight": FeatureConfig(
                    count=20,
                    parameters={},
                    placement_rules={"spacing": 30.0, "offset_from_road": 5.0, "side": "both"}
                ),
                "utility_cabinet": FeatureConfig(
                    count=4,
                    parameters={},
                    placement_rules={"offset_from_road": 8.0, "side": "both"}
                ),
                "speed_hump": FeatureConfig(
                    count=0,  # No speed humps on highways
                    parameters={},
                    placement_rules={}
                )
            },
            composition_rules={
                "multi_lane": True,
                "median_barrier": True,
                "shoulder_safety": True,
                "high_speed_design": True
            },
            output_settings={
                "format": "laz",
                "compression": True,
                "include_metadata": True
            }
        )
        
        # Park Configuration
        configs["urban_park"] = SceneConfig(
            name="Urban Park",
            description="City park with paths, benches, trees, and recreational facilities",
            complexity=SceneComplexity.DETAILED,
            scene_size={"width": 100.0, "length": 100.0, "height": 20.0},
            features={
                "sidewalk": FeatureConfig(
                    count=4,
                    parameters={"width": 2.0, "length": 100.0},
                    placement_rules={"grid_pattern": True, "spacing": 25.0}
                ),
                "tree": FeatureConfig(
                    count=50,
                    parameters={},
                    placement_rules={"natural_clustering": True, "min_spacing": 5.0}
                ),
                "bench": FeatureConfig(
                    count=15,
                    parameters={},
                    placement_rules={"near_paths": True, "spacing": 15.0}
                ),
                "picnic_table": FeatureConfig(
                    count=8,
                    parameters={},
                    placement_rules={"grouped": True, "spacing": 20.0}
                ),
                "trash_can": FeatureConfig(
                    count=12,
                    parameters={},
                    placement_rules={"near_benches": True, "spacing": 25.0}
                ),
                "bike_rack_u": FeatureConfig(
                    count=6,
                    parameters={},
                    placement_rules={"near_entrances": True}
                ),
                "streetlight": FeatureConfig(
                    count=20,
                    parameters={},
                    placement_rules={"path_lighting": True, "spacing": 20.0}
                ),
                "fire_hydrant": FeatureConfig(
                    count=4,
                    parameters={},
                    placement_rules={"perimeter": True}
                )
            },
            composition_rules={
                "natural_layout": True,
                "recreational_focus": True,
                "pedestrian_priority": True,
                "green_space": True
            },
            output_settings={
                "format": "laz",
                "compression": True,
                "include_metadata": True
            }
        )
        
        return configs
    
    def get_predefined_config(self, config_name: str) -> SceneConfig:
        """
        Get a predefined configuration by name.
        
        Args:
            config_name: Name of the predefined configuration
            
        Returns:
            SceneConfig object
        """
        if config_name not in self.predefined_configs:
            available = list(self.predefined_configs.keys())
            raise ValueError(f"Unknown configuration '{config_name}'. Available: {available}")
        
        return self.predefined_configs[config_name]
    
    def get_available_configs(self) -> List[str]:
        """Get list of available predefined configurations."""
        return list(self.predefined_configs.keys())
    
    def create_custom_config(self, name: str, description: str, 
                           features: Dict[str, FeatureConfig],
                           scene_size: Dict[str, float] = None,
                           complexity: SceneComplexity = SceneComplexity.STANDARD) -> SceneConfig:
        """
        Create a custom scene configuration.
        
        Args:
            name: Configuration name
            description: Configuration description
            features: Dictionary of feature configurations
            scene_size: Scene dimensions
            complexity: Scene complexity level
            
        Returns:
            SceneConfig object
        """
        if scene_size is None:
            scene_size = {"width": 50.0, "length": 50.0, "height": 15.0}
        
        return SceneConfig(
            name=name,
            description=description,
            complexity=complexity,
            scene_size=scene_size,
            features=features,
            composition_rules={},
            output_settings={
                "format": "laz",
                "compression": True,
                "include_metadata": True
            }
        )
    
    def save_config(self, config: SceneConfig, filepath: Path):
        """
        Save a configuration to a JSON file.
        
        Args:
            config: SceneConfig to save
            filepath: Path to save the configuration
        """
        config_dict = asdict(config)
        # Convert enum to string for JSON serialization
        config_dict["complexity"] = config.complexity.value
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self, filepath: Path) -> SceneConfig:
        """
        Load a configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            SceneConfig object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert string back to enum
        config_dict["complexity"] = SceneComplexity(config_dict["complexity"])
        
        # Reconstruct FeatureConfig objects
        features = {}
        for name, feature_dict in config_dict["features"].items():
            features[name] = FeatureConfig(**feature_dict)
        
        config_dict["features"] = features
        
        return SceneConfig(**config_dict)
    
    def get_config_summary(self, config_name: str) -> Dict[str, Any]:
        """
        Get a summary of a configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Dictionary with configuration summary
        """
        config = self.get_predefined_config(config_name)
        
        total_features = sum(feature.count for feature in config.features.values() if feature.enabled)
        feature_types = len([f for f in config.features.values() if f.enabled])
        
        return {
            "name": config.name,
            "description": config.description,
            "complexity": config.complexity.value,
            "scene_size": config.scene_size,
            "total_features": total_features,
            "feature_types": feature_types,
            "enabled_features": [name for name, feature in config.features.items() if feature.enabled]
        }
