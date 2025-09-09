#!/usr/bin/env python3
"""
Large-Scale Synthetic Scene Generation

This script provides professional, modular generation of large-scale synthetic
LiDAR point cloud scenes with realistic urban infrastructure composition.

Features:
- Configurable scene types (residential, commercial, highway, park)
- Realistic feature placement following urban planning principles
- Batch generation with multiple scene variations
- Professional output with metadata and documentation
- Command-line interface with configuration options

Usage:
    python generate_large_scale_scenes.py --config residential_street --output_dir ./output
    python generate_large_scale_scenes.py --custom-config my_config.json --count 5
    python generate_large_scale_scenes.py --list-configs
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# Import our modular components
from synthetic_scenes import SceneGenerator
from scene_config import SceneConfigurationManager, SceneConfig, SceneComplexity
from scene_composer import SceneComposer

# Import LAS writing functionality
from generate_point_cloud_sandbox import write_las, ADD_RGB, WRITE_LAZ


class LargeScaleGenerator:
    """
    Professional large-scale synthetic scene generator.
    
    Provides batch generation of realistic urban scenes with configurable
    feature selection and realistic composition.
    """
    
    def __init__(self, output_dir: Path = None, config_manager: SceneConfigurationManager = None):
        """
        Initialize the large-scale generator.
        
        Args:
            output_dir: Directory for output files
            config_manager: Configuration manager instance
        """
        self.output_dir = output_dir or Path("large_scale_output")
        self.config_manager = config_manager or SceneConfigurationManager()
        self.scene_generator = SceneGenerator()
        self.scene_composer = SceneComposer(self.scene_generator)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_scene(self, config: SceneConfig, scene_name: str = None) -> Dict[str, Any]:
        """
        Generate a single scene from configuration.
        
        Args:
            config: Scene configuration
            scene_name: Optional custom name for the scene
            
        Returns:
            Dictionary with generation results and metadata
        """
        if scene_name is None:
            scene_name = config.name.lower().replace(" ", "_")
        
        print(f"Generating scene: {config.name}")
        print(f"  Complexity: {config.complexity.value}")
        print(f"  Scene size: {config.scene_size['width']:.1f}m x {config.scene_size['length']:.1f}m")
        print(f"  Features: {sum(f.count for f in config.features.values() if f.enabled)} total")
        
        start_time = time.time()
        
        try:
            # Compose the scene
            scene_data = self.scene_composer.compose_scene(config)
            
            # Generate output filename
            timestamp = int(time.time())
            filename = f"{scene_name}_{timestamp}"
            
            # Write LAS/LAZ file
            las_file = self.output_dir / f"{filename}.las"
            write_las(las_file, scene_data)
            
            # Convert to LAZ if requested
            if WRITE_LAZ:
                try:
                    import laspy
                    laz_file = self.output_dir / f"{filename}.laz"
                    las = laspy.read(las_file)
                    las.write(laz_file)
                    las_file.unlink()  # Remove temporary LAS file
                    final_file = laz_file
                except Exception as e:
                    print(f"  Warning: Could not compress to LAZ ({e}), keeping LAS file")
                    final_file = las_file
            else:
                final_file = las_file
            
            # Generate metadata
            metadata = self._generate_metadata(config, scene_data, final_file)
            
            # Save metadata
            metadata_file = self.output_dir / f"{filename}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            generation_time = time.time() - start_time
            
            print(f"  ✓ Generated {final_file.name}")
            print(f"  ✓ Points: {len(scene_data['x']):,}")
            print(f"  ✓ Time: {generation_time:.1f}s")
            
            return {
                'success': True,
                'scene_name': scene_name,
                'output_file': final_file,
                'metadata_file': metadata_file,
                'point_count': len(scene_data['x']),
                'generation_time': generation_time,
                'metadata': metadata,
                'scene_data': scene_data  # Include scene data for preview
            }
            
        except Exception as e:
            print(f"  ✗ Error generating scene: {e}")
            return {
                'success': False,
                'scene_name': scene_name,
                'error': str(e)
            }
    
    def generate_batch(self, config: SceneConfig, count: int = 1, 
                      base_name: str = None) -> List[Dict[str, Any]]:
        """
        Generate multiple scenes from the same configuration.
        
        Args:
            config: Scene configuration
            count: Number of scenes to generate
            base_name: Base name for generated scenes
            
        Returns:
            List of generation results
        """
        if base_name is None:
            base_name = config.name.lower().replace(" ", "_")
        
        print(f"Generating batch: {count} scenes of type '{config.name}'")
        print(f"Output directory: {self.output_dir}")
        print()
        
        results = []
        for i in range(count):
            scene_name = f"{base_name}_{i+1:03d}"
            result = self.generate_scene(config, scene_name)
            results.append(result)
            print()
        
        # Generate batch summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"Batch complete: {len(successful)}/{count} successful")
        if failed:
            print(f"Failed scenes: {[r['scene_name'] for r in failed]}")
        
        return results
    
    def generate_from_custom_config(self, config_file: Path, count: int = 1) -> List[Dict[str, Any]]:
        """
        Generate scenes from a custom configuration file.
        
        Args:
            config_file: Path to custom configuration JSON file
            count: Number of scenes to generate
            
        Returns:
            List of generation results
        """
        try:
            config = self.config_manager.load_config(config_file)
            return self.generate_batch(config, count)
        except Exception as e:
            print(f"Error loading custom configuration: {e}")
            return []
    
    def _generate_metadata(self, config: SceneConfig, scene_data: Dict[str, np.ndarray], 
                          output_file: Path) -> Dict[str, Any]:
        """Generate comprehensive metadata for a scene."""
        # Calculate scene statistics
        point_count = len(scene_data['x'])
        unique_classes, class_counts = np.unique(scene_data['cls'], return_counts=True)
        
        # Calculate bounding box
        bounds = {
            'min_x': float(np.min(scene_data['x'])),
            'max_x': float(np.max(scene_data['x'])),
            'min_y': float(np.min(scene_data['y'])),
            'max_y': float(np.max(scene_data['y'])),
            'min_z': float(np.min(scene_data['z'])),
            'max_z': float(np.max(scene_data['z']))
        }
        
        # Calculate scene dimensions
        dimensions = {
            'width': bounds['max_x'] - bounds['min_x'],
            'length': bounds['max_y'] - bounds['min_y'],
            'height': bounds['max_z'] - bounds['min_z']
        }
        
        # Calculate point density
        area = dimensions['width'] * dimensions['length']
        density = point_count / area if area > 0 else 0
        
        # Feature summary
        enabled_features = {name: feature.count for name, feature in config.features.items() 
                          if feature.enabled}
        
        return {
            'generation_info': {
                'timestamp': time.time(),
                'generator_version': '1.0.0',
                'config_name': config.name,
                'config_description': config.description,
                'complexity': config.complexity.value
            },
            'scene_info': {
                'output_file': output_file.name,
                'point_count': point_count,
                'bounds': bounds,
                'dimensions': dimensions,
                'point_density_per_sqm': density,
                'has_rgb': ADD_RGB,
                'format': 'LAZ' if WRITE_LAZ else 'LAS'
            },
            'features': {
                'enabled_features': enabled_features,
                'total_feature_count': sum(enabled_features.values()),
                'feature_types': len(enabled_features)
            },
            'classification_summary': {
                'unique_classes': unique_classes.tolist(),
                'class_counts': dict(zip(unique_classes.tolist(), class_counts.tolist()))
            },
            'configuration': {
                'scene_size': config.scene_size,
                'composition_rules': config.composition_rules,
                'output_settings': config.output_settings
            }
        }
    
    def preview_scene(self, scene_data: Dict[str, np.ndarray]):
        """
        Open 3D preview of the generated scene.
        
        Args:
            scene_data: Point cloud data to preview
        """
        try:
            import open3d as o3d
            
            # Create point cloud
            points = np.vstack([scene_data['x'], scene_data['y'], scene_data['z']]).T
            
            # Create colors
            if ADD_RGB and all(k in scene_data for k in ["red", "green", "blue"]):
                colors = np.vstack([
                    scene_data['red'],
                    scene_data['green'],
                    scene_data['blue']
                ]).T.astype(np.float32) / 65535.0
            else:
                # Use intensity for grayscale
                colors = (scene_data['intensity'].astype(np.float32) / 65535.0).reshape(-1, 1)
                colors = np.repeat(colors, 3, axis=1)
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            print("  Opening 3D preview...")
            print("  Close the preview window to continue...")
            
            # Use the same visualization as the original demo_scene.py
            o3d.visualization.draw_geometries([pcd], 
                                            window_name="Synthetic Scene Preview",
                                            width=1024, 
                                            height=768)
            
        except ImportError as e:
            print(f"  Warning: Open3D not available for preview: {e}")
            print("  Install Open3D with: pip install open3d")
        except Exception as e:
            print(f"  Warning: Could not open preview: {e}")
            import traceback
            traceback.print_exc()
    
    def list_available_configs(self):
        """List all available predefined configurations."""
        configs = self.config_manager.get_available_configs()
        
        print("Available predefined configurations:")
        print()
        
        for config_name in configs:
            summary = self.config_manager.get_config_summary(config_name)
            print(f"  {config_name}")
            print(f"    Description: {summary['description']}")
            print(f"    Complexity: {summary['complexity']}")
            print(f"    Size: {summary['scene_size']['width']:.0f}m x {summary['scene_size']['length']:.0f}m")
            print(f"    Features: {summary['total_features']} total, {summary['feature_types']} types")
            print()


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate large-scale synthetic LiDAR scenes with realistic urban infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a residential street scene
  python generate_large_scale_scenes.py --config residential_street

  # Generate 5 commercial intersection scenes
  python generate_large_scale_scenes.py --config commercial_intersection --count 5

  # Use custom configuration file
  python generate_large_scale_scenes.py --custom-config my_scene.json --count 3

  # List available configurations
  python generate_large_scale_scenes.py --list-configs

  # Generate with custom output directory
  python generate_large_scale_scenes.py --config urban_park --output-dir ./my_scenes
  
  # Generate and preview the scene
  python generate_large_scale_scenes.py --config residential_street --preview
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument(
        '--config', '-c',
        help='Predefined configuration name'
    )
    config_group.add_argument(
        '--custom-config',
        type=Path,
        help='Path to custom configuration JSON file'
    )
    config_group.add_argument(
        '--list-configs',
        action='store_true',
        help='List available predefined configurations'
    )
    
    # Generation options
    parser.add_argument(
        '--count', '-n',
        type=int,
        default=1,
        help='Number of scenes to generate (default: 1)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('large_scale_output'),
        help='Output directory for generated scenes (default: large_scale_output)'
    )
    
    # Advanced options
    parser.add_argument(
        '--format',
        choices=['las', 'laz'],
        default='laz',
        help='Output format (default: laz)'
    )
    
    parser.add_argument(
        '--no-rgb',
        action='store_true',
        help='Generate intensity-only data (no RGB colors)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Open 3D preview after generation'
    )
    
    args = parser.parse_args()
    
    # Handle list-configs option
    if args.list_configs:
        generator = LargeScaleGenerator()
        generator.list_available_configs()
        return
    
    # Validate arguments
    if not args.config and not args.custom_config:
        parser.error("Must specify either --config or --custom-config")
    
    if args.count < 1:
        parser.error("Count must be at least 1")
    
    # Set global options
    global WRITE_LAZ, ADD_RGB
    WRITE_LAZ = (args.format == 'laz')
    ADD_RGB = not args.no_rgb
    
    # Initialize generator
    generator = LargeScaleGenerator(output_dir=args.output_dir)
    
    try:
        # Generate scenes
        if args.config:
            # Use predefined configuration
            config = generator.config_manager.get_predefined_config(args.config)
            results = generator.generate_batch(config, args.count)
        else:
            # Use custom configuration
            results = generator.generate_from_custom_config(args.custom_config, args.count)
        
        # Handle preview for successful generations
        if args.preview:
            successful_results = [r for r in results if r['success']]
            if successful_results:
                # Preview the first successful result
                first_result = successful_results[0]
                if 'scene_data' in first_result:
                    generator.preview_scene(first_result['scene_data'])
                else:
                    print("  Warning: Scene data not available for preview")
            else:
                print("  Warning: No successful generations to preview")
        
        # Print summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\nGeneration Summary:")
        print(f"  Total scenes: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        if successful:
            total_points = sum(r['point_count'] for r in successful)
            total_time = sum(r['generation_time'] for r in successful)
            print(f"  Total points: {total_points:,}")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Output directory: {args.output_dir}")
        
        if failed:
            print(f"\nFailed scenes:")
            for result in failed:
                print(f"  {result['scene_name']}: {result['error']}")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
