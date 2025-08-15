"""
XR Spatial Computing Engine for Photonic Neuromorphic Systems.

This module provides advanced spatial computing capabilities for XR applications,
including 3D scene understanding, object persistence, and spatial AI processing
using photonic neural networks.
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import json

from .core import PhotonicSNN, WaveguideNeuron, OpticalParameters
from .xr_agent_mesh import XRCoordinate, XRDataType, XRMessage
from .monitoring import MetricsCollector
from .exceptions import ValidationError, OpticalModelError


@dataclass
class SpatialObject:
    """Represents a spatial object in XR space."""
    object_id: str
    position: XRCoordinate
    bounding_box: Dict[str, float]  # {'width': x, 'height': y, 'depth': z}
    object_class: str
    confidence: float
    semantic_features: List[float]
    last_updated: float = field(default_factory=lambda: datetime.now().timestamp())
    persistence_score: float = 1.0
    
    def get_center_point(self) -> np.ndarray:
        """Get center point of the object."""
        return np.array([self.position.x, self.position.y, self.position.z])
    
    def get_corners(self) -> np.ndarray:
        """Get 8 corner points of the bounding box."""
        center = self.get_center_point()
        w, h, d = self.bounding_box['width'], self.bounding_box['height'], self.bounding_box['depth']
        
        # Generate 8 corners
        corners = np.array([
            [-w/2, -h/2, -d/2], [w/2, -h/2, -d/2], [w/2, h/2, -d/2], [-w/2, h/2, -d/2],
            [-w/2, -h/2, d/2], [w/2, -h/2, d/2], [w/2, h/2, d/2], [-w/2, h/2, d/2]
        ])
        
        # Apply rotation if needed
        if any([self.position.rotation_x, self.position.rotation_y, self.position.rotation_z]):
            rotation = Rotation.from_euler('xyz', [
                self.position.rotation_x, self.position.rotation_y, self.position.rotation_z
            ])
            corners = rotation.apply(corners)
        
        return corners + center
    
    def overlaps_with(self, other: 'SpatialObject', threshold: float = 0.1) -> bool:
        """Check if this object overlaps with another object."""
        distance = np.linalg.norm(self.get_center_point() - other.get_center_point())
        min_separation = (
            max(self.bounding_box['width'], self.bounding_box['height'], self.bounding_box['depth']) +
            max(other.bounding_box['width'], other.bounding_box['height'], other.bounding_box['depth'])
        ) / 2
        
        return distance < (min_separation + threshold)


@dataclass
class SpatialRegion:
    """Represents a spatial region with semantic meaning."""
    region_id: str
    boundary_points: List[XRCoordinate]
    region_type: str  # 'room', 'surface', 'boundary', etc.
    properties: Dict[str, Any]
    contained_objects: List[str] = field(default_factory=list)
    
    def contains_point(self, point: XRCoordinate) -> bool:
        """Check if a point is within this region (simple 2D polygon containment)."""
        if len(self.boundary_points) < 3:
            return False
        
        # Convert to 2D for simplicity (project onto XY plane)
        polygon = [(p.x, p.y) for p in self.boundary_points]
        test_point = (point.x, point.y)
        
        # Ray casting algorithm
        x, y = test_point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


class PhotonicSpatialProcessor:
    """Photonic neural processor specialized for spatial computing tasks."""
    
    def __init__(self):
        """Initialize photonic spatial processor."""
        # Specialized networks for different spatial tasks
        self.object_detection_network = PhotonicSNN(
            topology=[512, 1024, 512, 256],  # Object detection
            wavelength=1550e-9
        )
        
        self.spatial_reasoning_network = PhotonicSNN(
            topology=[256, 512, 256, 128],  # Spatial relationships
            wavelength=1310e-9  # Different wavelength for parallel processing
        )
        
        self.scene_understanding_network = PhotonicSNN(
            topology=[1024, 2048, 1024, 512],  # Scene context
            wavelength=1270e-9
        )
        
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
        
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for monitoring."""
        self._metrics_collector = collector
        self.object_detection_network.set_metrics_collector(collector)
        self.spatial_reasoning_network.set_metrics_collector(collector)
        self.scene_understanding_network.set_metrics_collector(collector)
    
    async def process_object_detection(self, sensor_data: Dict[str, Any]) -> List[SpatialObject]:
        """Process sensor data for object detection."""
        try:
            # Encode sensor data (depth, RGB, point cloud, etc.)
            input_features = self._encode_sensor_data(sensor_data)
            
            # Convert to spike train
            from .core import encode_to_spikes
            spike_train = encode_to_spikes(input_features, duration=100e-9)
            
            # Process through object detection network
            detection_output = self.object_detection_network(spike_train)
            
            # Decode to spatial objects
            objects = self._decode_object_detections(detection_output, sensor_data)
            
            if self._metrics_collector:
                self._metrics_collector.record_metric("objects_detected", len(objects))
                self._metrics_collector.increment_counter("object_detection_runs")
            
            return objects
            
        except Exception as e:
            if self._metrics_collector:
                self._metrics_collector.increment_counter("object_detection_errors")
            self._logger.error(f"Object detection failed: {e}")
            return []
    
    async def process_spatial_relationships(self, objects: List[SpatialObject]) -> Dict[str, Any]:
        """Analyze spatial relationships between objects."""
        try:
            if len(objects) < 2:
                return {"relationships": []}
            
            # Create pairwise relationship features
            relationship_features = self._compute_relationship_features(objects)
            
            # Process through spatial reasoning network
            from .core import encode_to_spikes
            spike_train = encode_to_spikes(relationship_features, duration=75e-9)
            reasoning_output = self.spatial_reasoning_network(spike_train)
            
            # Decode relationships
            relationships = self._decode_spatial_relationships(reasoning_output, objects)
            
            if self._metrics_collector:
                self._metrics_collector.record_metric("spatial_relationships_found", len(relationships))
                self._metrics_collector.increment_counter("spatial_reasoning_runs")
            
            return {"relationships": relationships}
            
        except Exception as e:
            if self._metrics_collector:
                self._metrics_collector.increment_counter("spatial_reasoning_errors")
            self._logger.error(f"Spatial reasoning failed: {e}")
            return {"relationships": []}
    
    async def process_scene_understanding(self, objects: List[SpatialObject], 
                                        regions: List[SpatialRegion]) -> Dict[str, Any]:
        """Generate high-level scene understanding."""
        try:
            # Create scene context features
            scene_features = self._compute_scene_features(objects, regions)
            
            # Process through scene understanding network
            from .core import encode_to_spikes
            spike_train = encode_to_spikes(scene_features, duration=150e-9)
            scene_output = self.scene_understanding_network(spike_train)
            
            # Decode scene understanding
            scene_analysis = self._decode_scene_understanding(scene_output, objects, regions)
            
            if self._metrics_collector:
                self._metrics_collector.record_metric("scene_complexity", len(objects) + len(regions))
                self._metrics_collector.increment_counter("scene_understanding_runs")
            
            return scene_analysis
            
        except Exception as e:
            if self._metrics_collector:
                self._metrics_collector.increment_counter("scene_understanding_errors")
            self._logger.error(f"Scene understanding failed: {e}")
            return {"scene_type": "unknown", "confidence": 0.0}
    
    def _encode_sensor_data(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """Encode various sensor inputs into feature vector."""
        features = []
        
        # Process depth data
        if 'depth' in sensor_data:
            depth_data = np.array(sensor_data['depth']).flatten()
            # Downsample and normalize
            depth_features = depth_data[::max(1, len(depth_data)//128)][:128]
            features.extend(depth_features)
        
        # Process RGB data
        if 'rgb' in sensor_data:
            rgb_data = np.array(sensor_data['rgb']).flatten()
            # Extract color histograms or key features
            rgb_features = rgb_data[::max(1, len(rgb_data)//128)][:128]
            features.extend(rgb_features)
        
        # Process point cloud data
        if 'point_cloud' in sensor_data:
            points = np.array(sensor_data['point_cloud'])
            # Extract geometric features
            if len(points) > 0:
                point_features = [
                    np.mean(points, axis=0),  # Centroid
                    np.std(points, axis=0),   # Spread
                    np.min(points, axis=0),   # Bounding box min
                    np.max(points, axis=0)    # Bounding box max
                ].flatten()
                features.extend(point_features)
        
        # Process IMU data
        if 'imu' in sensor_data:
            imu_data = sensor_data['imu']
            imu_features = [
                imu_data.get('acceleration', [0, 0, 0]),
                imu_data.get('gyroscope', [0, 0, 0]),
                imu_data.get('magnetometer', [0, 0, 0])
            ]
            features.extend(np.array(imu_features).flatten())
        
        # Pad or truncate to fixed size
        target_size = 512
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _decode_object_detections(self, neural_output: torch.Tensor, 
                                sensor_data: Dict[str, Any]) -> List[SpatialObject]:
        """Decode neural network output to spatial objects."""
        # Sum spikes across time dimension
        output_features = torch.sum(neural_output, dim=0).detach().numpy()
        
        objects = []
        detection_threshold = 0.1
        
        # Assume output is organized as: [class_scores, positions, sizes, ...]
        # This is a simplified decoding - would be more sophisticated in practice
        for i in range(0, len(output_features) - 9, 10):  # 10 features per object
            class_score = output_features[i]
            
            if class_score > detection_threshold:
                position = XRCoordinate(
                    x=float(output_features[i+1]),
                    y=float(output_features[i+2]),
                    z=float(output_features[i+3])
                )
                
                bounding_box = {
                    'width': max(0.1, float(output_features[i+4])),
                    'height': max(0.1, float(output_features[i+5])),
                    'depth': max(0.1, float(output_features[i+6]))
                }
                
                obj = SpatialObject(
                    object_id=f"obj_{i//10}_{datetime.now().timestamp():.6f}",
                    position=position,
                    bounding_box=bounding_box,
                    object_class=f"class_{int(output_features[i+7]) % 10}",
                    confidence=float(class_score),
                    semantic_features=output_features[i+8:i+10].tolist()
                )
                
                objects.append(obj)
        
        return objects
    
    def _compute_relationship_features(self, objects: List[SpatialObject]) -> np.ndarray:
        """Compute pairwise relationship features between objects."""
        if len(objects) < 2:
            return np.zeros(256)
        
        features = []
        
        # Compute pairwise features
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                
                # Distance features
                distance = np.linalg.norm(obj1.get_center_point() - obj2.get_center_point())
                features.append(distance)
                
                # Relative position
                rel_pos = obj1.get_center_point() - obj2.get_center_point()
                features.extend(rel_pos.tolist())
                
                # Size comparison
                size1 = max(obj1.bounding_box.values())
                size2 = max(obj2.bounding_box.values())
                features.append(size1 / (size2 + 1e-8))
                
                # Overlap check
                features.append(1.0 if obj1.overlaps_with(obj2) else 0.0)
        
        # Pad or truncate to fixed size
        target_size = 256
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _decode_spatial_relationships(self, neural_output: torch.Tensor, 
                                    objects: List[SpatialObject]) -> List[Dict[str, Any]]:
        """Decode neural output to spatial relationships."""
        output_features = torch.sum(neural_output, dim=0).detach().numpy()
        
        relationships = []
        relationship_threshold = 0.15
        
        # Simple relationship detection
        for i in range(0, len(output_features) - 4, 5):
            relationship_strength = output_features[i]
            
            if relationship_strength > relationship_threshold:
                # Decode relationship type and objects
                obj1_idx = int(output_features[i+1] * len(objects)) % len(objects)
                obj2_idx = int(output_features[i+2] * len(objects)) % len(objects)
                
                if obj1_idx != obj2_idx:
                    relationship_type_code = int(output_features[i+3] * 5) % 5
                    relationship_types = ['above', 'below', 'near', 'inside', 'adjacent']
                    
                    relationships.append({
                        'object1': objects[obj1_idx].object_id,
                        'object2': objects[obj2_idx].object_id,
                        'relationship_type': relationship_types[relationship_type_code],
                        'confidence': float(relationship_strength),
                        'distance': float(output_features[i+4])
                    })
        
        return relationships
    
    def _compute_scene_features(self, objects: List[SpatialObject], 
                              regions: List[SpatialRegion]) -> np.ndarray:
        """Compute scene-level features."""
        features = []
        
        # Object statistics
        features.append(len(objects))
        features.append(len(regions))
        
        if objects:
            # Object distribution
            positions = np.array([obj.get_center_point() for obj in objects])
            features.extend([
                np.mean(positions[:, 0]),  # Average X
                np.mean(positions[:, 1]),  # Average Y
                np.mean(positions[:, 2]),  # Average Z
                np.std(positions[:, 0]),   # Spread X
                np.std(positions[:, 1]),   # Spread Y
                np.std(positions[:, 2])    # Spread Z
            ])
            
            # Object classes histogram
            class_counts = {}
            for obj in objects:
                class_counts[obj.object_class] = class_counts.get(obj.object_class, 0) + 1
            
            # Top 10 classes
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            class_features = [count for _, count in sorted_classes]
            class_features.extend([0] * (10 - len(class_features)))
            features.extend(class_features)
        else:
            features.extend([0.0] * 16)  # Empty scene features
        
        # Region features
        if regions:
            region_sizes = [len(region.boundary_points) for region in regions]
            features.extend([
                np.mean(region_sizes),
                np.std(region_sizes),
                max(region_sizes),
                min(region_sizes)
            ])
        else:
            features.extend([0.0] * 4)
        
        # Pad to target size
        target_size = 1024
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _decode_scene_understanding(self, neural_output: torch.Tensor, 
                                  objects: List[SpatialObject], 
                                  regions: List[SpatialRegion]) -> Dict[str, Any]:
        """Decode neural output to scene understanding."""
        output_features = torch.sum(neural_output, dim=0).detach().numpy()
        
        # Scene classification
        scene_types = ['indoor', 'outdoor', 'office', 'home', 'industrial', 'retail', 'public']
        scene_scores = output_features[:len(scene_types)]
        predicted_scene = scene_types[np.argmax(scene_scores)]
        scene_confidence = float(np.max(scene_scores))
        
        # Activity prediction
        activities = ['static', 'navigation', 'manipulation', 'social', 'work', 'play']
        activity_scores = output_features[len(scene_types):len(scene_types)+len(activities)]
        if len(activity_scores) == len(activities):
            predicted_activity = activities[np.argmax(activity_scores)]
            activity_confidence = float(np.max(activity_scores))
        else:
            predicted_activity = 'unknown'
            activity_confidence = 0.0
        
        return {
            'scene_type': predicted_scene,
            'scene_confidence': scene_confidence,
            'predicted_activity': predicted_activity,
            'activity_confidence': activity_confidence,
            'scene_complexity': len(objects) + len(regions),
            'object_density': len(objects) / max(len(regions), 1),
            'spatial_extent': self._compute_spatial_extent(objects)
        }
    
    def _compute_spatial_extent(self, objects: List[SpatialObject]) -> Dict[str, float]:
        """Compute spatial extent of the scene."""
        if not objects:
            return {'x_range': 0.0, 'y_range': 0.0, 'z_range': 0.0, 'volume': 0.0}
        
        positions = np.array([obj.get_center_point() for obj in objects])
        
        x_range = float(np.max(positions[:, 0]) - np.min(positions[:, 0]))
        y_range = float(np.max(positions[:, 1]) - np.min(positions[:, 1]))
        z_range = float(np.max(positions[:, 2]) - np.min(positions[:, 2]))
        volume = x_range * y_range * z_range
        
        return {
            'x_range': x_range,
            'y_range': y_range,
            'z_range': z_range,
            'volume': volume
        }


class SpatialMemoryManager:
    """Manages persistent spatial memory for XR environments."""
    
    def __init__(self, memory_capacity: int = 10000):
        """Initialize spatial memory manager."""
        self.memory_capacity = memory_capacity
        self.objects: Dict[str, SpatialObject] = {}
        self.regions: Dict[str, SpatialRegion] = {}
        self.spatial_index = None  # KDTree for spatial queries
        self.persistence_threshold = 0.1
        
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
        
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for monitoring."""
        self._metrics_collector = collector
    
    def add_object(self, obj: SpatialObject):
        """Add or update object in spatial memory."""
        # Check for existing similar objects
        existing_obj = self._find_similar_object(obj)
        
        if existing_obj:
            # Update existing object
            self._merge_objects(existing_obj, obj)
            if self._metrics_collector:
                self._metrics_collector.increment_counter("spatial_objects_updated")
        else:
            # Add new object
            self.objects[obj.object_id] = obj
            if self._metrics_collector:
                self._metrics_collector.increment_counter("spatial_objects_added")
        
        # Rebuild spatial index
        self._rebuild_spatial_index()
        
        # Cleanup if memory is full
        if len(self.objects) > self.memory_capacity:
            self._cleanup_old_objects()
    
    def add_region(self, region: SpatialRegion):
        """Add spatial region to memory."""
        self.regions[region.region_id] = region
        
        if self._metrics_collector:
            self._metrics_collector.increment_counter("spatial_regions_added")
    
    def query_objects_in_radius(self, center: XRCoordinate, radius: float) -> List[SpatialObject]:
        """Query objects within radius of a point."""
        if not self.spatial_index or not self.objects:
            return []
        
        center_point = np.array([center.x, center.y, center.z])
        
        # Query spatial index
        indices = self.spatial_index.query_ball_point(center_point, radius)
        
        # Return corresponding objects
        object_list = list(self.objects.values())
        return [object_list[i] for i in indices if i < len(object_list)]
    
    def query_objects_by_class(self, object_class: str) -> List[SpatialObject]:
        """Query objects by class."""
        return [obj for obj in self.objects.values() if obj.object_class == object_class]
    
    def get_containing_region(self, point: XRCoordinate) -> Optional[SpatialRegion]:
        """Find region containing the given point."""
        for region in self.regions.values():
            if region.contains_point(point):
                return region
        return None
    
    def update_object_persistence(self, object_id: str, seen: bool = True):
        """Update object persistence score based on observations."""
        if object_id in self.objects:
            obj = self.objects[object_id]
            if seen:
                obj.persistence_score = min(1.0, obj.persistence_score + 0.1)
                obj.last_updated = datetime.now().timestamp()
            else:
                obj.persistence_score = max(0.0, obj.persistence_score - 0.05)
            
            # Remove if persistence too low
            if obj.persistence_score < self.persistence_threshold:
                del self.objects[object_id]
                self._rebuild_spatial_index()
                if self._metrics_collector:
                    self._metrics_collector.increment_counter("spatial_objects_removed")
    
    def _find_similar_object(self, new_obj: SpatialObject) -> Optional[SpatialObject]:
        """Find existing object similar to the new one."""
        similarity_threshold = 1.0  # 1 meter
        
        for existing_obj in self.objects.values():
            if (existing_obj.object_class == new_obj.object_class and
                existing_obj.position.distance_to(new_obj.position) < similarity_threshold):
                return existing_obj
        
        return None
    
    def _merge_objects(self, existing_obj: SpatialObject, new_obj: SpatialObject):
        """Merge new object information with existing object."""
        # Update position (weighted average based on confidence)
        total_confidence = existing_obj.confidence + new_obj.confidence
        weight_existing = existing_obj.confidence / total_confidence
        weight_new = new_obj.confidence / total_confidence
        
        existing_obj.position.x = (existing_obj.position.x * weight_existing + 
                                 new_obj.position.x * weight_new)
        existing_obj.position.y = (existing_obj.position.y * weight_existing + 
                                 new_obj.position.y * weight_new)
        existing_obj.position.z = (existing_obj.position.z * weight_existing + 
                                 new_obj.position.z * weight_new)
        
        # Update confidence (take maximum)
        existing_obj.confidence = max(existing_obj.confidence, new_obj.confidence)
        
        # Update timestamp
        existing_obj.last_updated = datetime.now().timestamp()
        
        # Increase persistence score
        existing_obj.persistence_score = min(1.0, existing_obj.persistence_score + 0.2)
    
    def _rebuild_spatial_index(self):
        """Rebuild KDTree spatial index."""
        if not self.objects:
            self.spatial_index = None
            return
        
        points = np.array([obj.get_center_point() for obj in self.objects.values()])
        self.spatial_index = KDTree(points)
    
    def _cleanup_old_objects(self):
        """Remove old objects to maintain memory capacity."""
        # Sort by persistence score and last update time
        objects_list = list(self.objects.items())
        objects_list.sort(key=lambda x: (x[1].persistence_score, -x[1].last_updated))
        
        # Remove lowest scoring objects
        num_to_remove = len(objects_list) - self.memory_capacity + 100  # Remove extra for buffer
        
        for i in range(min(num_to_remove, len(objects_list))):
            object_id = objects_list[i][0]
            del self.objects[object_id]
            
        self._rebuild_spatial_index()
        
        if self._metrics_collector:
            self._metrics_collector.record_metric("spatial_objects_cleaned", num_to_remove)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get spatial memory statistics."""
        if not self.objects:
            return {
                'total_objects': 0,
                'total_regions': len(self.regions),
                'memory_usage': 0.0,
                'avg_persistence': 0.0
            }
        
        persistence_scores = [obj.persistence_score for obj in self.objects.values()]
        
        return {
            'total_objects': len(self.objects),
            'total_regions': len(self.regions),
            'memory_usage': len(self.objects) / self.memory_capacity,
            'avg_persistence': np.mean(persistence_scores),
            'min_persistence': np.min(persistence_scores),
            'max_persistence': np.max(persistence_scores)
        }
    
    def export_spatial_map(self, filepath: str):
        """Export spatial map to file."""
        map_data = {
            'objects': {
                obj_id: {
                    'position': {
                        'x': obj.position.x,
                        'y': obj.position.y,
                        'z': obj.position.z,
                        'rotation_x': obj.position.rotation_x,
                        'rotation_y': obj.position.rotation_y,
                        'rotation_z': obj.position.rotation_z
                    },
                    'bounding_box': obj.bounding_box,
                    'object_class': obj.object_class,
                    'confidence': obj.confidence,
                    'persistence_score': obj.persistence_score,
                    'semantic_features': obj.semantic_features
                }
                for obj_id, obj in self.objects.items()
            },
            'regions': {
                region_id: {
                    'boundary_points': [
                        {'x': p.x, 'y': p.y, 'z': p.z} 
                        for p in region.boundary_points
                    ],
                    'region_type': region.region_type,
                    'properties': region.properties,
                    'contained_objects': region.contained_objects
                }
                for region_id, region in self.regions.items()
            },
            'metadata': {
                'export_timestamp': datetime.now().timestamp(),
                'memory_capacity': self.memory_capacity,
                'total_objects': len(self.objects),
                'total_regions': len(self.regions)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        self._logger.info(f"Exported spatial map to {filepath}")


def create_spatial_computing_demo() -> Dict[str, Any]:
    """Create a demonstration of spatial computing capabilities."""
    processor = PhotonicSpatialProcessor()
    memory_manager = SpatialMemoryManager()
    
    # Set up metrics
    metrics_collector = MetricsCollector()
    processor.set_metrics_collector(metrics_collector)
    memory_manager.set_metrics_collector(metrics_collector)
    
    # Simulate sensor data
    demo_sensor_data = {
        'depth': np.random.rand(64, 64),  # 64x64 depth image
        'rgb': np.random.rand(64, 64, 3),  # RGB image
        'point_cloud': np.random.rand(1000, 3),  # 1000 3D points
        'imu': {
            'acceleration': [0.1, 0.0, 9.8],
            'gyroscope': [0.0, 0.0, 0.1],
            'magnetometer': [0.3, 0.0, 0.9]
        }
    }
    
    return {
        'processor': processor,
        'memory_manager': memory_manager,
        'demo_sensor_data': demo_sensor_data,
        'metrics_collector': metrics_collector
    }


async def run_spatial_computing_demo() -> Dict[str, Any]:
    """Run spatial computing demonstration."""
    demo = create_spatial_computing_demo()
    processor = demo['processor']
    memory_manager = demo['memory_manager']
    sensor_data = demo['demo_sensor_data']
    
    try:
        # Object detection
        detected_objects = await processor.process_object_detection(sensor_data)
        
        # Add objects to spatial memory
        for obj in detected_objects:
            memory_manager.add_object(obj)
        
        # Spatial relationship analysis
        relationships = await processor.process_spatial_relationships(detected_objects)
        
        # Scene understanding
        scene_analysis = await processor.process_scene_understanding(detected_objects, [])
        
        # Spatial queries
        query_center = XRCoordinate(0, 0, 0)
        nearby_objects = memory_manager.query_objects_in_radius(query_center, 5.0)
        
        memory_stats = memory_manager.get_memory_stats()
        
        return {
            'detected_objects': len(detected_objects),
            'spatial_relationships': relationships,
            'scene_analysis': scene_analysis,
            'nearby_objects_count': len(nearby_objects),
            'memory_stats': memory_stats,
            'metrics_summary': demo['metrics_collector'].get_metrics_summary()
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'detected_objects': 0,
            'metrics_summary': demo['metrics_collector'].get_metrics_summary()
        }