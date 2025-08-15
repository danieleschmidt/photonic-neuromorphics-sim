"""
XR Visualization and Interaction Engine for Photonic Neuromorphic Systems.

This module provides real-time XR visualization, interaction handling, and
haptic feedback processing using photonic neural networks for ultra-low
latency responses.
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json
import threading
import queue

from .core import PhotonicSNN, WaveguideNeuron
from .xr_agent_mesh import XRCoordinate, XRDataType, XRMessage, XRAgent
from .xr_spatial_computing import SpatialObject, SpatialRegion, PhotonicSpatialProcessor
from .monitoring import MetricsCollector
from .exceptions import ValidationError, OpticalModelError


class InteractionType(Enum):
    """Types of XR interactions."""
    GAZE = "gaze"
    GESTURE = "gesture"
    VOICE = "voice"
    HAPTIC = "haptic"
    PROXIMITY = "proximity"
    COLLISION = "collision"


class RenderingMode(Enum):
    """XR rendering modes."""
    WIREFRAME = "wireframe"
    SOLID = "solid"
    TEXTURED = "textured"
    HOLOGRAPHIC = "holographic"


@dataclass
class XRInteraction:
    """Represents an XR interaction event."""
    interaction_id: str
    interaction_type: InteractionType
    position: XRCoordinate
    target_object_id: Optional[str] = None
    gesture_data: Optional[Dict[str, Any]] = None
    intensity: float = 1.0
    duration: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert interaction to feature vector for neural processing."""
        features = [
            self.position.x, self.position.y, self.position.z,
            self.position.rotation_x, self.position.rotation_y, self.position.rotation_z,
            float(self.interaction_type.value.__hash__() % 100) / 100.0,
            self.intensity,
            self.duration
        ]
        
        # Add gesture data if available
        if self.gesture_data:
            gesture_features = []
            for key in ['velocity', 'acceleration', 'direction']:
                if key in self.gesture_data:
                    gesture_features.extend(self.gesture_data[key][:3])  # Take first 3 components
                else:
                    gesture_features.extend([0.0, 0.0, 0.0])
            features.extend(gesture_features)
        else:
            features.extend([0.0] * 9)  # Empty gesture features
        
        return np.array(features, dtype=np.float32)


@dataclass
class HapticFeedback:
    """Represents haptic feedback data."""
    feedback_id: str
    position: XRCoordinate
    force_vector: Tuple[float, float, float]
    vibration_frequency: float = 0.0
    vibration_amplitude: float = 0.0
    texture_roughness: float = 0.0
    temperature_delta: float = 0.0
    duration: float = 0.1
    
    def to_actuation_signals(self) -> Dict[str, float]:
        """Convert to haptic actuator signals."""
        return {
            'force_x': self.force_vector[0],
            'force_y': self.force_vector[1],
            'force_z': self.force_vector[2],
            'vibration_freq': self.vibration_frequency,
            'vibration_amp': self.vibration_amplitude,
            'texture': self.texture_roughness,
            'thermal': self.temperature_delta
        }


class PhotonicInteractionProcessor:
    """Photonic neural processor for XR interaction handling."""
    
    def __init__(self):
        """Initialize photonic interaction processor."""
        # Specialized networks for different interaction modalities
        self.gesture_recognition_network = PhotonicSNN(
            topology=[256, 512, 256, 64],  # Gesture classification
            wavelength=1550e-9
        )
        
        self.haptic_generation_network = PhotonicSNN(
            topology=[128, 256, 128, 32],  # Haptic feedback generation
            wavelength=1310e-9
        )
        
        self.interaction_prediction_network = PhotonicSNN(
            topology=[384, 768, 384, 96],  # Interaction intent prediction
            wavelength=1270e-9
        )
        
        # Response time tracking
        self.processing_times = {
            'gesture_recognition': [],
            'haptic_generation': [],
            'interaction_prediction': []
        }
        
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
        
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for monitoring."""
        self._metrics_collector = collector
        self.gesture_recognition_network.set_metrics_collector(collector)
        self.haptic_generation_network.set_metrics_collector(collector)
        self.interaction_prediction_network.set_metrics_collector(collector)
    
    async def process_gesture_recognition(self, interaction: XRInteraction) -> Dict[str, Any]:
        """Process gesture recognition through photonic neural network."""
        try:
            start_time = datetime.now().timestamp()
            
            # Encode interaction data
            input_features = interaction.to_feature_vector()
            
            # Pad to network input size
            if len(input_features) < 256:
                input_features = np.pad(input_features, (0, 256 - len(input_features)))
            else:
                input_features = input_features[:256]
            
            # Convert to spike train
            from .core import encode_to_spikes
            spike_train = encode_to_spikes(input_features, duration=25e-9)  # Ultra-fast 25ns
            
            # Process through gesture recognition network
            gesture_output = self.gesture_recognition_network(spike_train)
            
            # Decode gesture
            gesture_result = self._decode_gesture_output(gesture_output)
            
            # Track processing time
            processing_time = datetime.now().timestamp() - start_time
            self.processing_times['gesture_recognition'].append(processing_time)
            
            if self._metrics_collector:
                self._metrics_collector.record_metric("gesture_processing_time", processing_time)
                self._metrics_collector.increment_counter("gestures_processed")
            
            return gesture_result
            
        except Exception as e:
            if self._metrics_collector:
                self._metrics_collector.increment_counter("gesture_processing_errors")
            self._logger.error(f"Gesture recognition failed: {e}")
            return {'gesture_type': 'unknown', 'confidence': 0.0}
    
    async def process_haptic_generation(self, interaction: XRInteraction, 
                                      context_objects: List[SpatialObject]) -> HapticFeedback:
        """Generate haptic feedback based on interaction and context."""
        try:
            start_time = datetime.now().timestamp()
            
            # Create context features
            context_features = self._create_haptic_context(interaction, context_objects)
            
            # Convert to spike train
            from .core import encode_to_spikes
            spike_train = encode_to_spikes(context_features, duration=15e-9)  # Ultra-fast 15ns
            
            # Process through haptic generation network
            haptic_output = self.haptic_generation_network(spike_train)
            
            # Decode haptic feedback
            haptic_feedback = self._decode_haptic_output(haptic_output, interaction)
            
            # Track processing time
            processing_time = datetime.now().timestamp() - start_time
            self.processing_times['haptic_generation'].append(processing_time)
            
            if self._metrics_collector:
                self._metrics_collector.record_metric("haptic_processing_time", processing_time)
                self._metrics_collector.increment_counter("haptic_feedback_generated")
            
            return haptic_feedback
            
        except Exception as e:
            if self._metrics_collector:
                self._metrics_collector.increment_counter("haptic_generation_errors")
            self._logger.error(f"Haptic generation failed: {e}")
            return HapticFeedback(
                feedback_id=f"error_{datetime.now().timestamp()}",
                position=interaction.position,
                force_vector=(0.0, 0.0, 0.0)
            )
    
    async def predict_user_intent(self, interaction_history: List[XRInteraction]) -> Dict[str, Any]:
        """Predict user intent from interaction history."""
        try:
            start_time = datetime.now().timestamp()
            
            # Create temporal features from interaction history
            temporal_features = self._create_temporal_features(interaction_history)
            
            # Convert to spike train
            from .core import encode_to_spikes
            spike_train = encode_to_spikes(temporal_features, duration=50e-9)
            
            # Process through interaction prediction network
            prediction_output = self.interaction_prediction_network(spike_train)
            
            # Decode intent prediction
            intent_result = self._decode_intent_output(prediction_output)
            
            # Track processing time
            processing_time = datetime.now().timestamp() - start_time
            self.processing_times['interaction_prediction'].append(processing_time)
            
            if self._metrics_collector:
                self._metrics_collector.record_metric("intent_prediction_time", processing_time)
                self._metrics_collector.increment_counter("intent_predictions_made")
            
            return intent_result
            
        except Exception as e:
            if self._metrics_collector:
                self._metrics_collector.increment_counter("intent_prediction_errors")
            self._logger.error(f"Intent prediction failed: {e}")
            return {'predicted_intent': 'unknown', 'confidence': 0.0}
    
    def _decode_gesture_output(self, neural_output: torch.Tensor) -> Dict[str, Any]:
        """Decode neural output to gesture classification."""
        output_features = torch.sum(neural_output, dim=0).detach().numpy()
        
        # Define gesture classes
        gesture_classes = [
            'point', 'grab', 'release', 'swipe_left', 'swipe_right',
            'swipe_up', 'swipe_down', 'pinch', 'spread', 'tap',
            'double_tap', 'circle', 'wave', 'thumbs_up', 'idle'
        ]
        
        # Get gesture prediction
        gesture_scores = output_features[:len(gesture_classes)]
        if len(gesture_scores) < len(gesture_classes):
            gesture_scores = np.pad(gesture_scores, (0, len(gesture_classes) - len(gesture_scores)))
        
        predicted_gesture_idx = np.argmax(gesture_scores)
        confidence = float(gesture_scores[predicted_gesture_idx])
        
        # Extract gesture parameters
        parameters = {}
        if len(output_features) > len(gesture_classes):
            param_features = output_features[len(gesture_classes):]
            parameters = {
                'velocity': float(param_features[0]) if len(param_features) > 0 else 0.0,
                'direction': float(param_features[1]) if len(param_features) > 1 else 0.0,
                'size': float(param_features[2]) if len(param_features) > 2 else 1.0
            }
        
        return {
            'gesture_type': gesture_classes[predicted_gesture_idx],
            'confidence': confidence,
            'parameters': parameters,
            'all_scores': gesture_scores.tolist()
        }
    
    def _create_haptic_context(self, interaction: XRInteraction, 
                             context_objects: List[SpatialObject]) -> np.ndarray:
        """Create context features for haptic generation."""
        features = []
        
        # Interaction features
        interaction_features = interaction.to_feature_vector()
        features.extend(interaction_features)
        
        # Object context features
        if context_objects:
            # Find closest object
            closest_object = min(
                context_objects,
                key=lambda obj: interaction.position.distance_to(obj.position)
            )
            
            # Distance to closest object
            distance = interaction.position.distance_to(closest_object.position)
            features.append(distance)
            
            # Object material properties (simulated)
            material_properties = self._get_material_properties(closest_object.object_class)
            features.extend(material_properties)
            
            # Object size influence
            object_size = max(closest_object.bounding_box.values())
            features.append(object_size)
        else:
            # No objects nearby
            features.extend([10.0] + [0.0] * 6)  # Large distance, no material properties
        
        # Pad to network input size
        target_size = 128
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _get_material_properties(self, object_class: str) -> List[float]:
        """Get simulated material properties for haptic feedback."""
        material_db = {
            'metal': [0.8, 0.2, 0.9, 0.1, 0.0],  # hardness, roughness, conductivity, elasticity, temperature
            'wood': [0.5, 0.6, 0.1, 0.3, 0.0],
            'plastic': [0.4, 0.3, 0.2, 0.7, 0.0],
            'glass': [0.9, 0.1, 0.3, 0.1, 0.0],
            'fabric': [0.1, 0.8, 0.0, 0.9, 0.0],
            'default': [0.5, 0.5, 0.5, 0.5, 0.0]
        }
        
        return material_db.get(object_class, material_db['default'])
    
    def _decode_haptic_output(self, neural_output: torch.Tensor, 
                            interaction: XRInteraction) -> HapticFeedback:
        """Decode neural output to haptic feedback."""
        output_features = torch.sum(neural_output, dim=0).detach().numpy()
        
        # Extract haptic parameters
        if len(output_features) >= 7:
            force_x = float(output_features[0] * 10.0)  # Scale to reasonable force range
            force_y = float(output_features[1] * 10.0)
            force_z = float(output_features[2] * 10.0)
            vibration_freq = float(output_features[3] * 1000.0)  # Hz
            vibration_amp = float(output_features[4])
            texture_roughness = float(output_features[5])
            temperature_delta = float(output_features[6] * 10.0)  # Celsius
        else:
            # Default values if output is incomplete
            force_x = force_y = force_z = 0.0
            vibration_freq = vibration_amp = texture_roughness = temperature_delta = 0.0
        
        return HapticFeedback(
            feedback_id=f"haptic_{datetime.now().timestamp():.6f}",
            position=interaction.position,
            force_vector=(force_x, force_y, force_z),
            vibration_frequency=vibration_freq,
            vibration_amplitude=vibration_amp,
            texture_roughness=texture_roughness,
            temperature_delta=temperature_delta,
            duration=0.1
        )
    
    def _create_temporal_features(self, interaction_history: List[XRInteraction]) -> np.ndarray:
        """Create temporal features from interaction history."""
        features = []
        
        # Take last 10 interactions
        recent_interactions = interaction_history[-10:] if len(interaction_history) > 10 else interaction_history
        
        for interaction in recent_interactions:
            interaction_features = interaction.to_feature_vector()
            features.extend(interaction_features[:10])  # Take first 10 features
        
        # Pad with zeros if not enough interactions
        while len(features) < 100:  # 10 interactions * 10 features
            features.extend([0.0] * 10)
        
        # Add temporal statistics
        if recent_interactions:
            # Time intervals between interactions
            time_intervals = []
            for i in range(1, len(recent_interactions)):
                interval = recent_interactions[i].timestamp - recent_interactions[i-1].timestamp
                time_intervals.append(interval)
            
            if time_intervals:
                features.extend([
                    np.mean(time_intervals),
                    np.std(time_intervals),
                    np.min(time_intervals),
                    np.max(time_intervals)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Position variance
            positions = np.array([[i.position.x, i.position.y, i.position.z] 
                                for i in recent_interactions])
            features.extend([
                np.std(positions[:, 0]),
                np.std(positions[:, 1]),
                np.std(positions[:, 2])
            ])
        else:
            features.extend([0.0] * 7)
        
        # Pad to target size
        target_size = 384
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return np.array(features, dtype=np.float32)
    
    def _decode_intent_output(self, neural_output: torch.Tensor) -> Dict[str, Any]:
        """Decode neural output to user intent prediction."""
        output_features = torch.sum(neural_output, dim=0).detach().numpy()
        
        # Define intent classes
        intent_classes = [
            'navigate', 'select', 'manipulate', 'create', 'delete',
            'annotate', 'measure', 'inspect', 'collaborate', 'idle'
        ]
        
        # Get intent prediction
        intent_scores = output_features[:len(intent_classes)]
        if len(intent_scores) < len(intent_classes):
            intent_scores = np.pad(intent_scores, (0, len(intent_classes) - len(intent_scores)))
        
        predicted_intent_idx = np.argmax(intent_scores)
        confidence = float(intent_scores[predicted_intent_idx])
        
        # Extract intent parameters
        parameters = {}
        if len(output_features) > len(intent_classes):
            param_features = output_features[len(intent_classes):]
            parameters = {
                'urgency': float(param_features[0]) if len(param_features) > 0 else 0.5,
                'precision_required': float(param_features[1]) if len(param_features) > 1 else 0.5,
                'collaborative': float(param_features[2]) if len(param_features) > 2 else 0.0
            }
        
        return {
            'predicted_intent': intent_classes[predicted_intent_idx],
            'confidence': confidence,
            'parameters': parameters,
            'next_likely_actions': self._get_likely_next_actions(intent_classes[predicted_intent_idx])
        }
    
    def _get_likely_next_actions(self, current_intent: str) -> List[str]:
        """Get likely next actions based on current intent."""
        action_transitions = {
            'navigate': ['select', 'inspect', 'measure'],
            'select': ['manipulate', 'delete', 'inspect'],
            'manipulate': ['create', 'annotate', 'select'],
            'create': ['manipulate', 'annotate', 'inspect'],
            'delete': ['select', 'navigate', 'create'],
            'annotate': ['inspect', 'select', 'navigate'],
            'measure': ['annotate', 'inspect', 'navigate'],
            'inspect': ['select', 'measure', 'annotate'],
            'collaborate': ['select', 'annotate', 'create'],
            'idle': ['navigate', 'select', 'inspect']
        }
        
        return action_transitions.get(current_intent, ['navigate', 'select'])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the interaction processor."""
        summary = {}
        
        for network_name, times in self.processing_times.items():
            if times:
                summary[network_name] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'std_time': np.std(times),
                    'total_processed': len(times)
                }
            else:
                summary[network_name] = {
                    'avg_time': 0.0,
                    'min_time': 0.0,
                    'max_time': 0.0,
                    'std_time': 0.0,
                    'total_processed': 0
                }
        
        return summary


class XRVisualizationEngine:
    """Real-time XR visualization engine with photonic neural processing."""
    
    def __init__(self, window_size: Tuple[int, int] = (800, 600)):
        """Initialize XR visualization engine."""
        self.window_size = window_size
        self.rendering_mode = RenderingMode.SOLID
        self.objects_to_render: List[SpatialObject] = []
        self.interaction_points: List[XRInteraction] = []
        self.haptic_feedback_points: List[HapticFeedback] = []
        
        # Visualization state
        self.camera_position = XRCoordinate(0, 0, 5)
        self.camera_target = XRCoordinate(0, 0, 0)
        self.field_of_view = 60.0
        
        # Real-time updates
        self.update_queue = queue.Queue()
        self.is_rendering = False
        self.frame_rate = 60.0
        self.last_frame_time = 0.0
        
        # Performance tracking
        self.render_times = []
        self.frame_count = 0
        
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
        
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for monitoring."""
        self._metrics_collector = collector
    
    def add_spatial_object(self, obj: SpatialObject):
        """Add spatial object to visualization."""
        self.objects_to_render.append(obj)
        
        if self._metrics_collector:
            self._metrics_collector.increment_counter("visualization_objects_added")
    
    def remove_spatial_object(self, object_id: str):
        """Remove spatial object from visualization."""
        self.objects_to_render = [obj for obj in self.objects_to_render 
                                if obj.object_id != object_id]
        
        if self._metrics_collector:
            self._metrics_collector.increment_counter("visualization_objects_removed")
    
    def add_interaction_point(self, interaction: XRInteraction):
        """Add interaction point to visualization."""
        self.interaction_points.append(interaction)
        
        # Keep only recent interactions (last 50)
        if len(self.interaction_points) > 50:
            self.interaction_points = self.interaction_points[-50:]
    
    def add_haptic_feedback(self, feedback: HapticFeedback):
        """Add haptic feedback visualization."""
        self.haptic_feedback_points.append(feedback)
        
        # Keep only recent feedback (last 20)
        if len(self.haptic_feedback_points) > 20:
            self.haptic_feedback_points = self.haptic_feedback_points[-20:]
    
    def set_camera_position(self, position: XRCoordinate, target: XRCoordinate = None):
        """Set camera position and target."""
        self.camera_position = position
        if target:
            self.camera_target = target
    
    def start_rendering(self):
        """Start real-time rendering."""
        self.is_rendering = True
        self.frame_count = 0
        self.last_frame_time = datetime.now().timestamp()
        
        # Start rendering in separate thread
        rendering_thread = threading.Thread(target=self._rendering_loop)
        rendering_thread.daemon = True
        rendering_thread.start()
        
        self._logger.info("XR visualization rendering started")
    
    def stop_rendering(self):
        """Stop real-time rendering."""
        self.is_rendering = False
        self._logger.info("XR visualization rendering stopped")
    
    def _rendering_loop(self):
        """Main rendering loop."""
        plt.ion()  # Interactive mode
        fig = plt.figure(figsize=(self.window_size[0]/100, self.window_size[1]/100))
        ax = fig.add_subplot(111, projection='3d')
        
        while self.is_rendering:
            try:
                frame_start_time = datetime.now().timestamp()
                
                # Clear and setup axes
                ax.clear()
                ax.set_xlim([-10, 10])
                ax.set_ylim([-10, 10])
                ax.set_zlim([-5, 15])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'XR Scene Visualization - Frame {self.frame_count}')
                
                # Render spatial objects
                self._render_spatial_objects(ax)
                
                # Render interaction points
                self._render_interactions(ax)
                
                # Render haptic feedback
                self._render_haptic_feedback(ax)
                
                # Set camera view
                ax.view_init(
                    elev=20,  # Elevation angle
                    azim=self.frame_count * 0.5  # Slowly rotating view
                )
                
                plt.draw()
                plt.pause(1.0 / self.frame_rate)
                
                # Track performance
                frame_time = datetime.now().timestamp() - frame_start_time
                self.render_times.append(frame_time)
                
                if len(self.render_times) > 100:
                    self.render_times = self.render_times[-100:]  # Keep last 100 frames
                
                self.frame_count += 1
                
                if self._metrics_collector and self.frame_count % 60 == 0:  # Every second
                    avg_frame_time = np.mean(self.render_times)
                    self._metrics_collector.record_metric("avg_frame_time", avg_frame_time)
                    self._metrics_collector.record_metric("fps", 1.0 / avg_frame_time)
                
            except Exception as e:
                self._logger.error(f"Rendering error: {e}")
                break
        
        plt.ioff()
    
    def _render_spatial_objects(self, ax):
        """Render spatial objects in 3D."""
        for obj in self.objects_to_render:
            # Get object corners
            corners = obj.get_corners()
            
            # Create 3D wireframe box
            if self.rendering_mode in [RenderingMode.WIREFRAME, RenderingMode.SOLID]:
                # Define the 12 edges of a cube
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                ]
                
                for edge in edges:
                    points = corners[edge]
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                            color=self._get_object_color(obj.object_class), 
                            alpha=0.7)
            
            # Add object label
            center = obj.get_center_point()
            ax.text(center[0], center[1], center[2], 
                   f"{obj.object_class}\n{obj.confidence:.2f}",
                   fontsize=8)
    
    def _render_interactions(self, ax):
        """Render interaction points."""
        for interaction in self.interaction_points:
            pos = interaction.position
            
            # Color based on interaction type
            color = self._get_interaction_color(interaction.interaction_type)
            
            # Size based on intensity
            size = max(20, interaction.intensity * 100)
            
            ax.scatter(pos.x, pos.y, pos.z, 
                      c=color, s=size, alpha=0.8, marker='o')
            
            # Add interaction label
            ax.text(pos.x, pos.y, pos.z + 0.2, 
                   interaction.interaction_type.value,
                   fontsize=6)
    
    def _render_haptic_feedback(self, ax):
        """Render haptic feedback visualizations."""
        for feedback in self.haptic_feedback_points:
            pos = feedback.position
            
            # Render force vector
            force = feedback.force_vector
            if max(abs(f) for f in force) > 0.1:  # Only show significant forces
                ax.quiver(pos.x, pos.y, pos.z,
                         force[0], force[1], force[2],
                         color='red', alpha=0.6, arrow_length_ratio=0.1)
            
            # Render vibration as pulsing sphere
            if feedback.vibration_amplitude > 0.1:
                vibration_size = 50 + feedback.vibration_amplitude * 100
                ax.scatter(pos.x, pos.y, pos.z,
                          c='orange', s=vibration_size, alpha=0.3, marker='o')
    
    def _get_object_color(self, object_class: str) -> str:
        """Get color for object class."""
        color_map = {
            'table': 'brown',
            'chair': 'blue',
            'person': 'green',
            'wall': 'gray',
            'door': 'orange',
            'window': 'cyan',
            'default': 'purple'
        }
        return color_map.get(object_class, color_map['default'])
    
    def _get_interaction_color(self, interaction_type: InteractionType) -> str:
        """Get color for interaction type."""
        color_map = {
            InteractionType.GAZE: 'yellow',
            InteractionType.GESTURE: 'green',
            InteractionType.VOICE: 'blue',
            InteractionType.HAPTIC: 'red',
            InteractionType.PROXIMITY: 'orange',
            InteractionType.COLLISION: 'magenta'
        }
        return color_map.get(interaction_type, 'black')
    
    def capture_frame(self, filename: str = None) -> str:
        """Capture current frame to file."""
        if filename is None:
            filename = f"xr_frame_{datetime.now().timestamp():.6f}.png"
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        self._logger.info(f"Captured frame to {filename}")
        
        return filename
    
    def get_rendering_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics."""
        if not self.render_times:
            return {
                'frame_count': self.frame_count,
                'avg_frame_time': 0.0,
                'avg_fps': 0.0,
                'target_fps': self.frame_rate
            }
        
        avg_frame_time = np.mean(self.render_times)
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return {
            'frame_count': self.frame_count,
            'avg_frame_time': avg_frame_time,
            'min_frame_time': np.min(self.render_times),
            'max_frame_time': np.max(self.render_times),
            'avg_fps': avg_fps,
            'target_fps': self.frame_rate,
            'objects_rendered': len(self.objects_to_render),
            'interactions_shown': len(self.interaction_points),
            'haptic_points_shown': len(self.haptic_feedback_points)
        }


async def run_xr_visualization_demo(duration: float = 5.0) -> Dict[str, Any]:
    """Run XR visualization demonstration."""
    # Create demo components
    interaction_processor = PhotonicInteractionProcessor()
    visualization_engine = XRVisualizationEngine()
    
    # Set up metrics
    metrics_collector = MetricsCollector()
    interaction_processor.set_metrics_collector(metrics_collector)
    visualization_engine.set_metrics_collector(metrics_collector)
    
    # Create demo scene
    demo_objects = [
        SpatialObject(
            object_id="table_1",
            position=XRCoordinate(0, 0, 0),
            bounding_box={'width': 2.0, 'height': 0.8, 'depth': 1.0},
            object_class="table",
            confidence=0.9,
            semantic_features=[0.8, 0.2, 0.7]
        ),
        SpatialObject(
            object_id="chair_1",
            position=XRCoordinate(1.5, 0, 0.5),
            bounding_box={'width': 0.6, 'height': 1.0, 'depth': 0.6},
            object_class="chair",
            confidence=0.85,
            semantic_features=[0.5, 0.8, 0.6]
        )
    ]
    
    # Add objects to visualization
    for obj in demo_objects:
        visualization_engine.add_spatial_object(obj)
    
    try:
        # Start visualization
        visualization_engine.start_rendering()
        
        # Simulate interactions
        start_time = datetime.now().timestamp()
        interaction_count = 0
        
        while datetime.now().timestamp() - start_time < duration:
            # Generate random interaction
            interaction = XRInteraction(
                interaction_id=f"demo_interaction_{interaction_count}",
                interaction_type=InteractionType.GESTURE,
                position=XRCoordinate(
                    np.random.uniform(-3, 3),
                    np.random.uniform(-3, 3),
                    np.random.uniform(0, 3)
                ),
                gesture_data={
                    'velocity': [np.random.uniform(-1, 1) for _ in range(3)],
                    'acceleration': [np.random.uniform(-0.5, 0.5) for _ in range(3)]
                },
                intensity=np.random.uniform(0.3, 1.0)
            )
            
            # Process interaction
            gesture_result = await interaction_processor.process_gesture_recognition(interaction)
            haptic_feedback = await interaction_processor.process_haptic_generation(
                interaction, demo_objects
            )
            
            # Add to visualization
            visualization_engine.add_interaction_point(interaction)
            visualization_engine.add_haptic_feedback(haptic_feedback)
            
            interaction_count += 1
            await asyncio.sleep(0.1)  # 10 interactions per second
        
        # Stop visualization
        await asyncio.sleep(1.0)  # Let it render a bit more
        visualization_engine.stop_rendering()
        
        # Get results
        performance_summary = interaction_processor.get_performance_summary()
        rendering_stats = visualization_engine.get_rendering_stats()
        
        return {
            'demo_duration': duration,
            'interactions_processed': interaction_count,
            'interaction_performance': performance_summary,
            'rendering_stats': rendering_stats,
            'metrics_summary': metrics_collector.get_metrics_summary()
        }
        
    except Exception as e:
        visualization_engine.stop_rendering()
        return {
            'error': str(e),
            'demo_duration': duration,
            'metrics_summary': metrics_collector.get_metrics_summary()
        }


def create_interaction_demo_sequence() -> List[XRInteraction]:
    """Create a demonstration sequence of XR interactions."""
    interactions = []
    base_time = datetime.now().timestamp()
    
    # Gaze sequence
    for i in range(5):
        interactions.append(XRInteraction(
            interaction_id=f"gaze_{i}",
            interaction_type=InteractionType.GAZE,
            position=XRCoordinate(i * 0.5, 0, 1.5),
            intensity=0.8,
            timestamp=base_time + i * 0.2
        ))
    
    # Gesture sequence
    gesture_types = [InteractionType.GESTURE] * 5
    gesture_positions = [
        XRCoordinate(0, 0, 1),
        XRCoordinate(1, 0, 1),
        XRCoordinate(1, 1, 1),
        XRCoordinate(0, 1, 1),
        XRCoordinate(0, 0, 1)
    ]
    
    for i, (gtype, pos) in enumerate(zip(gesture_types, gesture_positions)):
        interactions.append(XRInteraction(
            interaction_id=f"gesture_{i}",
            interaction_type=gtype,
            position=pos,
            gesture_data={
                'velocity': [1.0, 0.0, 0.0],
                'acceleration': [0.5, 0.0, 0.0],
                'direction': [1.0, 0.0, 0.0]
            },
            intensity=0.9,
            duration=0.3,
            timestamp=base_time + 1.0 + i * 0.3
        ))
    
    return interactions