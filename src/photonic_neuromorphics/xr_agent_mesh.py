"""
XR Agent Mesh System for Photonic Neuromorphic Networks.

This module implements a distributed Extended Reality (XR) agent mesh system that leverages
photonic neuromorphic computing for ultra-low latency, high-bandwidth agent coordination
and spatial computing.
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np
import torch
from pydantic import BaseModel, Field, validator

from .core import PhotonicSNN, WaveguideNeuron, OpticalParameters
from .monitoring import MetricsCollector
from .exceptions import ValidationError, OpticalModelError


class XRAgentType(Enum):
    """Types of XR agents in the mesh."""
    SPATIAL_ANCHOR = "spatial_anchor"
    OBJECT_TRACKER = "object_tracker"
    ENVIRONMENT_MAPPER = "environment_mapper"
    INTERACTION_HANDLER = "interaction_handler"
    PHYSICS_SIMULATOR = "physics_simulator"
    RENDERING_COORDINATOR = "rendering_coordinator"
    COLLABORATION_MANAGER = "collaboration_manager"


class XRDataType(Enum):
    """Types of data flowing through the XR mesh."""
    SPATIAL_COORDINATES = "spatial_coordinates"
    OBJECT_DETECTION = "object_detection"
    GESTURE_RECOGNITION = "gesture_recognition"
    HAPTIC_FEEDBACK = "haptic_feedback"
    AUDIO_SPATIAL = "audio_spatial"
    VISUAL_TRACKING = "visual_tracking"
    ENVIRONMENT_MAP = "environment_map"
    PHYSICS_STATE = "physics_state"


@dataclass
class XRCoordinate:
    """3D coordinate with orientation for XR space."""
    x: float
    y: float
    z: float
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    confidence: float = 1.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to 6D vector [x, y, z, rx, ry, rz]."""
        return np.array([self.x, self.y, self.z, 
                        self.rotation_x, self.rotation_y, self.rotation_z])
    
    def distance_to(self, other: 'XRCoordinate') -> float:
        """Calculate 3D distance to another coordinate."""
        return np.sqrt((self.x - other.x)**2 + 
                      (self.y - other.y)**2 + 
                      (self.z - other.z)**2)


class XRMessage(BaseModel):
    """Message format for XR agent communication."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcast
    data_type: XRDataType
    payload: Dict[str, Any]
    coordinates: Optional[XRCoordinate] = None
    priority: int = Field(default=5, ge=1, le=10)  # 1=highest, 10=lowest
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    ttl: int = Field(default=100)  # Time to live in hops
    
    class Config:
        arbitrary_types_allowed = True


class PhotonicXRProcessor:
    """Photonic neural processor for XR data."""
    
    def __init__(self, 
                 input_dimensions: int = 512,
                 output_dimensions: int = 256,
                 processing_layers: List[int] = None,
                 wavelength: float = 1550e-9):
        """Initialize photonic XR processor."""
        if processing_layers is None:
            processing_layers = [input_dimensions, 1024, 512, output_dimensions]
        
        self.photonic_network = PhotonicSNN(
            topology=processing_layers,
            neuron_type=WaveguideNeuron,
            synapse_type="phase_change",
            wavelength=wavelength
        )
        
        self.input_encoder = self._create_input_encoder(input_dimensions)
        self.output_decoder = self._create_output_decoder(output_dimensions)
        self.processing_time = 0.0
        self.energy_consumption = 0.0
        
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
        
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for monitoring."""
        self._metrics_collector = collector
        self.photonic_network.set_metrics_collector(collector)
    
    def _create_input_encoder(self, dimensions: int) -> Callable:
        """Create input encoding function for XR data."""
        def encoder(xr_data: Dict[str, Any]) -> torch.Tensor:
            # Convert XR data to neural encoding
            if 'coordinates' in xr_data:
                coord = xr_data['coordinates']
                spatial_vector = coord.to_vector() if coord else np.zeros(6)
            else:
                spatial_vector = np.zeros(6)
            
            # Add object/gesture features
            object_features = np.array(xr_data.get('features', [0.0] * (dimensions - 6)))
            
            # Combine and normalize
            input_vector = np.concatenate([spatial_vector, object_features[:dimensions-6]])
            input_vector = input_vector[:dimensions]  # Ensure correct size
            
            # Pad if necessary
            if len(input_vector) < dimensions:
                input_vector = np.pad(input_vector, (0, dimensions - len(input_vector)))
            
            return torch.tensor(input_vector, dtype=torch.float32)
        
        return encoder
    
    def _create_output_decoder(self, dimensions: int) -> Callable:
        """Create output decoding function for XR results."""
        def decoder(neural_output: torch.Tensor) -> Dict[str, Any]:
            output_array = neural_output.detach().numpy()
            
            # Decode spatial components
            spatial_data = output_array[:6] if len(output_array) >= 6 else output_array
            
            # Decode feature components
            feature_data = output_array[6:] if len(output_array) > 6 else []
            
            return {
                'spatial_result': spatial_data.tolist(),
                'features': feature_data.tolist(),
                'confidence': float(np.mean(np.abs(output_array)))
            }
        
        return decoder
    
    async def process_xr_data(self, xr_message: XRMessage) -> Dict[str, Any]:
        """Process XR data through photonic neural network."""
        try:
            start_time = datetime.now().timestamp()
            
            # Encode input data
            input_tensor = self.input_encoder(xr_message.payload)
            
            # Convert to spike train
            from .core import encode_to_spikes
            spike_train = encode_to_spikes(input_tensor.numpy(), duration=50e-9)
            
            # Process through photonic network
            output_spikes = self.photonic_network(spike_train)
            
            # Decode output
            # Sum spikes across time to get final output
            output_tensor = torch.sum(output_spikes, dim=0)
            result = self.output_decoder(output_tensor)
            
            # Track performance
            self.processing_time = datetime.now().timestamp() - start_time
            energy_estimate = self.photonic_network.estimate_energy_consumption(spike_train)
            self.energy_consumption = energy_estimate['total_energy']
            
            if self._metrics_collector:
                self._metrics_collector.record_metric("xr_processing_time", self.processing_time)
                self._metrics_collector.record_metric("xr_energy_consumption", self.energy_consumption)
                self._metrics_collector.increment_counter("xr_messages_processed")
            
            self._logger.debug(f"Processed XR message {xr_message.id} in {self.processing_time*1e6:.1f} μs")
            
            return result
            
        except Exception as e:
            if self._metrics_collector:
                self._metrics_collector.increment_counter("xr_processing_errors")
            self._logger.error(f"XR processing failed for message {xr_message.id}: {e}")
            raise OpticalModelError("xr_processor", "process_xr_data", xr_message.payload, str(e))


class XRAgent(ABC):
    """Abstract base class for XR agents in the mesh."""
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: XRAgentType,
                 position: XRCoordinate,
                 processing_capability: Optional[PhotonicXRProcessor] = None):
        """Initialize XR agent."""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.position = position
        self.processor = processing_capability or PhotonicXRProcessor()
        self.neighbors: Dict[str, 'XRAgent'] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.is_active = False
        self.performance_metrics = {
            'messages_processed': 0,
            'avg_processing_time': 0.0,
            'total_energy_consumed': 0.0,
            'error_count': 0
        }
        
        self._logger = logging.getLogger(f"{__name__}.{agent_id}")
        self._metrics_collector = None
        
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for monitoring."""
        self._metrics_collector = collector
        self.processor.set_metrics_collector(collector)
    
    @abstractmethod
    async def process_local_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent-specific local data."""
        pass
    
    @abstractmethod
    async def handle_mesh_message(self, message: XRMessage) -> Optional[XRMessage]:
        """Handle incoming message from mesh."""
        pass
    
    async def send_message(self, message: XRMessage):
        """Send message to the mesh."""
        if not self.is_active:
            self._logger.warning(f"Agent {self.agent_id} not active, dropping message")
            return
        
        # Add to local queue for processing
        await self.message_queue.put(message)
        
        if self._metrics_collector:
            self._metrics_collector.increment_counter("xr_messages_sent")
    
    async def receive_message(self, message: XRMessage):
        """Receive message from another agent."""
        if not self.is_active:
            return
        
        try:
            # Process through photonic neural network
            processing_result = await self.processor.process_xr_data(message)
            
            # Handle with agent-specific logic
            response = await self.handle_mesh_message(message)
            
            # Update performance metrics
            self.performance_metrics['messages_processed'] += 1
            self.performance_metrics['total_energy_consumed'] += self.processor.energy_consumption
            
            if response:
                await self.send_message(response)
                
        except Exception as e:
            self.performance_metrics['error_count'] += 1
            self._logger.error(f"Error processing message {message.id}: {e}")
            
            if self._metrics_collector:
                self._metrics_collector.increment_counter("xr_message_processing_errors")
    
    async def start(self):
        """Start the agent."""
        self.is_active = True
        self._logger.info(f"XR Agent {self.agent_id} started at position {self.position}")
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
    
    async def stop(self):
        """Stop the agent."""
        self.is_active = False
        self._logger.info(f"XR Agent {self.agent_id} stopped")
    
    async def _message_processing_loop(self):
        """Main message processing loop."""
        while self.is_active:
            try:
                # Process queued messages
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self.receive_message(message)
                else:
                    await asyncio.sleep(0.001)  # 1ms sleep to prevent busy waiting
                    
            except Exception as e:
                self._logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(0.01)  # Brief pause on error
    
    def add_neighbor(self, neighbor: 'XRAgent'):
        """Add neighboring agent."""
        self.neighbors[neighbor.agent_id] = neighbor
        self._logger.debug(f"Added neighbor {neighbor.agent_id} to {self.agent_id}")
    
    def remove_neighbor(self, agent_id: str):
        """Remove neighboring agent."""
        if agent_id in self.neighbors:
            del self.neighbors[agent_id]
            self._logger.debug(f"Removed neighbor {agent_id} from {self.agent_id}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if self.performance_metrics['messages_processed'] > 0:
            avg_energy = self.performance_metrics['total_energy_consumed'] / self.performance_metrics['messages_processed']
        else:
            avg_energy = 0.0
            
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'position': self.position,
            'neighbors_count': len(self.neighbors),
            'performance': self.performance_metrics,
            'avg_energy_per_message': avg_energy,
            'error_rate': self.performance_metrics['error_count'] / max(self.performance_metrics['messages_processed'], 1)
        }


class SpatialAnchorAgent(XRAgent):
    """Agent for managing spatial anchors in XR space."""
    
    def __init__(self, agent_id: str, position: XRCoordinate):
        super().__init__(agent_id, XRAgentType.SPATIAL_ANCHOR, position)
        self.tracked_anchors: Dict[str, XRCoordinate] = {}
        self.anchor_update_threshold = 0.01  # 1cm threshold for updates
    
    async def process_local_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process spatial anchor data."""
        if 'new_anchor' in data:
            anchor_data = data['new_anchor']
            anchor_id = anchor_data.get('id', str(uuid.uuid4()))
            position = XRCoordinate(**anchor_data['position'])
            
            self.tracked_anchors[anchor_id] = position
            
            return {
                'action': 'anchor_registered',
                'anchor_id': anchor_id,
                'position': position
            }
        
        return {'action': 'no_action'}
    
    async def handle_mesh_message(self, message: XRMessage) -> Optional[XRMessage]:
        """Handle spatial anchor mesh messages."""
        if message.data_type == XRDataType.SPATIAL_COORDINATES:
            # Check if this updates any tracked anchors
            payload = message.payload
            
            if 'anchor_query' in payload:
                # Return known anchors near query position
                query_pos = XRCoordinate(**payload['position'])
                nearby_anchors = {}
                
                for anchor_id, anchor_pos in self.tracked_anchors.items():
                    distance = query_pos.distance_to(anchor_pos)
                    if distance < 5.0:  # 5m radius
                        nearby_anchors[anchor_id] = anchor_pos
                
                if nearby_anchors:
                    return XRMessage(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        data_type=XRDataType.SPATIAL_COORDINATES,
                        payload={
                            'anchor_response': nearby_anchors,
                            'query_id': payload.get('query_id')
                        }
                    )
        
        return None


class ObjectTrackerAgent(XRAgent):
    """Agent for tracking objects in XR space."""
    
    def __init__(self, agent_id: str, position: XRCoordinate):
        super().__init__(agent_id, XRAgentType.OBJECT_TRACKER, position)
        self.tracked_objects: Dict[str, Dict[str, Any]] = {}
        self.tracking_confidence_threshold = 0.7
    
    async def process_local_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process object tracking data."""
        if 'detected_objects' in data:
            objects = data['detected_objects']
            updated_objects = []
            
            for obj in objects:
                obj_id = obj.get('id', str(uuid.uuid4()))
                confidence = obj.get('confidence', 0.0)
                
                if confidence >= self.tracking_confidence_threshold:
                    self.tracked_objects[obj_id] = {
                        'position': XRCoordinate(**obj['position']),
                        'confidence': confidence,
                        'class': obj.get('class', 'unknown'),
                        'last_seen': datetime.now().timestamp()
                    }
                    updated_objects.append(obj_id)
            
            return {
                'action': 'objects_updated',
                'updated_objects': updated_objects,
                'total_tracked': len(self.tracked_objects)
            }
        
        return {'action': 'no_action'}
    
    async def handle_mesh_message(self, message: XRMessage) -> Optional[XRMessage]:
        """Handle object tracking mesh messages."""
        if message.data_type == XRDataType.OBJECT_DETECTION:
            payload = message.payload
            
            if 'object_query' in payload:
                query_class = payload.get('object_class', None)
                matching_objects = {}
                
                for obj_id, obj_data in self.tracked_objects.items():
                    if query_class is None or obj_data['class'] == query_class:
                        matching_objects[obj_id] = obj_data
                
                if matching_objects:
                    return XRMessage(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        data_type=XRDataType.OBJECT_DETECTION,
                        payload={
                            'object_response': matching_objects,
                            'query_id': payload.get('query_id')
                        }
                    )
        
        return None


class XRAgentMesh:
    """Distributed mesh network of XR agents using photonic neural processing."""
    
    def __init__(self, mesh_id: str = None):
        """Initialize XR agent mesh."""
        self.mesh_id = mesh_id or str(uuid.uuid4())
        self.agents: Dict[str, XRAgent] = {}
        self.topology: Dict[str, List[str]] = {}  # Agent ID -> neighbor IDs
        self.message_routing_table: Dict[str, str] = {}  # Destination -> next hop
        self.is_running = False
        
        # Performance tracking
        self.mesh_metrics = {
            'total_messages': 0,
            'avg_latency': 0.0,
            'network_utilization': 0.0,
            'error_rate': 0.0
        }
        
        self._logger = logging.getLogger(__name__)
        self._metrics_collector = None
        self._executor = ThreadPoolExecutor(max_workers=8)
    
    def set_metrics_collector(self, collector: MetricsCollector):
        """Set metrics collector for monitoring."""
        self._metrics_collector = collector
        
        # Propagate to all agents
        for agent in self.agents.values():
            agent.set_metrics_collector(collector)
    
    def add_agent(self, agent: XRAgent):
        """Add agent to the mesh."""
        self.agents[agent.agent_id] = agent
        self.topology[agent.agent_id] = []
        
        if self._metrics_collector:
            agent.set_metrics_collector(self._metrics_collector)
        
        self._logger.info(f"Added agent {agent.agent_id} to mesh {self.mesh_id}")
    
    def remove_agent(self, agent_id: str):
        """Remove agent from the mesh."""
        if agent_id in self.agents:
            # Stop the agent
            if self.agents[agent_id].is_active:
                asyncio.create_task(self.agents[agent_id].stop())
            
            # Remove from topology
            del self.agents[agent_id]
            del self.topology[agent_id]
            
            # Remove from other agents' neighbor lists
            for other_agent in self.agents.values():
                other_agent.remove_neighbor(agent_id)
            
            self._logger.info(f"Removed agent {agent_id} from mesh {self.mesh_id}")
    
    def connect_agents(self, agent1_id: str, agent2_id: str):
        """Create bidirectional connection between two agents."""
        if agent1_id in self.agents and agent2_id in self.agents:
            # Update topology
            if agent2_id not in self.topology[agent1_id]:
                self.topology[agent1_id].append(agent2_id)
            if agent1_id not in self.topology[agent2_id]:
                self.topology[agent2_id].append(agent1_id)
            
            # Update agent neighbor lists
            self.agents[agent1_id].add_neighbor(self.agents[agent2_id])
            self.agents[agent2_id].add_neighbor(self.agents[agent1_id])
            
            self._logger.debug(f"Connected agents {agent1_id} and {agent2_id}")
    
    def auto_connect_by_proximity(self, max_distance: float = 10.0):
        """Automatically connect agents within proximity threshold."""
        agent_list = list(self.agents.values())
        
        for i, agent1 in enumerate(agent_list):
            for agent2 in agent_list[i+1:]:
                distance = agent1.position.distance_to(agent2.position)
                if distance <= max_distance:
                    self.connect_agents(agent1.agent_id, agent2.agent_id)
    
    async def broadcast_message(self, message: XRMessage):
        """Broadcast message to all agents in the mesh."""
        message.ttl -= 1
        if message.ttl <= 0:
            self._logger.warning(f"Message {message.id} expired (TTL reached 0)")
            return
        
        # Send to all agents except sender
        for agent_id, agent in self.agents.items():
            if agent_id != message.sender_id and agent.is_active:
                await agent.receive_message(message)
        
        self.mesh_metrics['total_messages'] += 1
        
        if self._metrics_collector:
            self._metrics_collector.increment_counter("mesh_broadcast_messages")
    
    async def route_message(self, message: XRMessage, target_agent_id: str):
        """Route message to specific agent using shortest path."""
        if target_agent_id not in self.agents:
            self._logger.error(f"Target agent {target_agent_id} not found in mesh")
            return
        
        # Simple direct routing for now (can be enhanced with shortest path algorithms)
        target_agent = self.agents[target_agent_id]
        if target_agent.is_active:
            await target_agent.receive_message(message)
            
            if self._metrics_collector:
                self._metrics_collector.increment_counter("mesh_routed_messages")
    
    async def start_mesh(self):
        """Start all agents in the mesh."""
        self.is_running = True
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
        
        self._logger.info(f"Started XR mesh {self.mesh_id} with {len(self.agents)} agents")
        
        # Start mesh monitoring task
        asyncio.create_task(self._mesh_monitoring_loop())
    
    async def stop_mesh(self):
        """Stop all agents in the mesh."""
        self.is_running = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        self._logger.info(f"Stopped XR mesh {self.mesh_id}")
    
    async def _mesh_monitoring_loop(self):
        """Monitor mesh performance and health."""
        while self.is_running:
            try:
                # Collect mesh statistics
                active_agents = sum(1 for agent in self.agents.values() if agent.is_active)
                total_messages = sum(agent.performance_metrics['messages_processed'] 
                                   for agent in self.agents.values())
                total_errors = sum(agent.performance_metrics['error_count'] 
                                 for agent in self.agents.values())
                
                # Update mesh metrics
                self.mesh_metrics['network_utilization'] = active_agents / max(len(self.agents), 1)
                self.mesh_metrics['error_rate'] = total_errors / max(total_messages, 1)
                
                if self._metrics_collector:
                    self._metrics_collector.record_metric("mesh_active_agents", active_agents)
                    self._metrics_collector.record_metric("mesh_total_messages", total_messages)
                    self._metrics_collector.record_metric("mesh_error_rate", self.mesh_metrics['error_rate'])
                
                await asyncio.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                self._logger.error(f"Error in mesh monitoring: {e}")
                await asyncio.sleep(5.0)  # Longer pause on error
    
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get comprehensive mesh status."""
        agent_summaries = [agent.get_performance_summary() for agent in self.agents.values()]
        
        return {
            'mesh_id': self.mesh_id,
            'is_running': self.is_running,
            'agent_count': len(self.agents),
            'active_agents': sum(1 for agent in self.agents.values() if agent.is_active),
            'topology': self.topology,
            'mesh_metrics': self.mesh_metrics,
            'agents': agent_summaries
        }


def create_xr_demo_mesh() -> XRAgentMesh:
    """Create a demonstration XR agent mesh with various agent types."""
    mesh = XRAgentMesh("demo_xr_mesh")
    
    # Create spatial anchor agents
    anchor_agent_1 = SpatialAnchorAgent("anchor_1", XRCoordinate(0, 0, 0))
    anchor_agent_2 = SpatialAnchorAgent("anchor_2", XRCoordinate(5, 0, 0))
    
    # Create object tracker agents
    tracker_agent_1 = ObjectTrackerAgent("tracker_1", XRCoordinate(2.5, 2.5, 1))
    tracker_agent_2 = ObjectTrackerAgent("tracker_2", XRCoordinate(-2.5, 2.5, 1))
    
    # Add agents to mesh
    mesh.add_agent(anchor_agent_1)
    mesh.add_agent(anchor_agent_2)
    mesh.add_agent(tracker_agent_1)
    mesh.add_agent(tracker_agent_2)
    
    # Auto-connect by proximity
    mesh.auto_connect_by_proximity(max_distance=8.0)
    
    return mesh


async def run_xr_mesh_simulation(duration: float = 10.0) -> Dict[str, Any]:
    """Run XR mesh simulation for specified duration."""
    mesh = create_xr_demo_mesh()
    
    # Set up metrics collection
    metrics_collector = MetricsCollector()
    mesh.set_metrics_collector(metrics_collector)
    
    # Start the mesh
    await mesh.start_mesh()
    
    try:
        # Simulate XR activity
        start_time = datetime.now().timestamp()
        
        while datetime.now().timestamp() - start_time < duration:
            # Generate some test messages
            for agent_id, agent in mesh.agents.items():
                if agent.is_active and np.random.random() < 0.1:  # 10% chance per iteration
                    test_message = XRMessage(
                        sender_id=agent_id,
                        data_type=XRDataType.SPATIAL_COORDINATES,
                        payload={
                            'test_data': np.random.randn(10).tolist(),
                            'timestamp': datetime.now().timestamp()
                        }
                    )
                    await mesh.broadcast_message(test_message)
            
            await asyncio.sleep(0.1)  # 100ms simulation step
        
        # Get final status
        final_status = mesh.get_mesh_status()
        
        return {
            'simulation_duration': duration,
            'mesh_status': final_status,
            'metrics_summary': metrics_collector.get_metrics_summary()
        }
        
    finally:
        await mesh.stop_mesh()