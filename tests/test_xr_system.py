"""
Comprehensive tests for XR Agent Mesh System.

This module provides extensive testing for all XR components including:
- Agent mesh networking
- Spatial computing
- Visualization
- Reliability and fault tolerance
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from photonic_neuromorphics.xr_agent_mesh import (
    XRAgentMesh, XRAgent, SpatialAnchorAgent, ObjectTrackerAgent,
    PhotonicXRProcessor, XRCoordinate, XRMessage, XRDataType, XRAgentType,
    create_xr_demo_mesh
)
from photonic_neuromorphics.xr_spatial_computing import (
    PhotonicSpatialProcessor, SpatialMemoryManager, SpatialObject, SpatialRegion,
    create_spatial_computing_demo
)
from photonic_neuromorphics.xr_visualization import (
    PhotonicInteractionProcessor, XRVisualizationEngine, XRInteraction, HapticFeedback,
    InteractionType, RenderingMode, create_interaction_demo_sequence
)
from photonic_neuromorphics.xr_reliability import (
    HealthMonitor, SelfHealingManager, CircuitBreaker, RetryManager,
    ReliableXRAgent, FailureType, HealthStatus
)
from photonic_neuromorphics.monitoring import MetricsCollector


class TestXRCoordinate:
    """Test XR coordinate system."""
    
    def test_coordinate_creation(self):
        """Test XR coordinate creation."""
        coord = XRCoordinate(1.0, 2.0, 3.0)
        assert coord.x == 1.0
        assert coord.y == 2.0
        assert coord.z == 3.0
        assert coord.rotation_x == 0.0
        assert coord.rotation_y == 0.0
        assert coord.rotation_z == 0.0
    
    def test_coordinate_with_rotation(self):
        """Test XR coordinate with rotation."""
        coord = XRCoordinate(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        assert coord.rotation_x == 0.1
        assert coord.rotation_y == 0.2
        assert coord.rotation_z == 0.3
    
    def test_coordinate_to_vector(self):
        """Test coordinate to vector conversion."""
        coord = XRCoordinate(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        vector = coord.to_vector()
        expected = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        np.testing.assert_array_equal(vector, expected)
    
    def test_distance_calculation(self):
        """Test distance calculation between coordinates."""
        coord1 = XRCoordinate(0.0, 0.0, 0.0)
        coord2 = XRCoordinate(3.0, 4.0, 0.0)
        distance = coord1.distance_to(coord2)
        assert abs(distance - 5.0) < 1e-6  # 3-4-5 triangle


class TestXRMessage:
    """Test XR message system."""
    
    def test_message_creation(self):
        """Test XR message creation."""
        message = XRMessage(
            sender_id="test_sender",
            data_type=XRDataType.SPATIAL_COORDINATES,
            payload={"test": "data"}
        )
        
        assert message.sender_id == "test_sender"
        assert message.data_type == XRDataType.SPATIAL_COORDINATES
        assert message.payload == {"test": "data"}
        assert message.receiver_id is None
        assert message.priority == 5
        assert message.ttl == 100
    
    def test_message_with_coordinates(self):
        """Test message with coordinates."""
        coord = XRCoordinate(1.0, 2.0, 3.0)
        message = XRMessage(
            sender_id="test_sender",
            data_type=XRDataType.SPATIAL_COORDINATES,
            payload={"test": "data"},
            coordinates=coord
        )
        
        assert message.coordinates == coord


class TestPhotonicXRProcessor:
    """Test photonic XR processor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = PhotonicXRProcessor(
            input_dimensions=64,
            output_dimensions=32,
            processing_layers=[64, 128, 64, 32]
        )
    
    @pytest.mark.asyncio
    async def test_xr_data_processing(self):
        """Test XR data processing."""
        message = XRMessage(
            sender_id="test",
            data_type=XRDataType.SPATIAL_COORDINATES,
            payload={
                'coordinates': XRCoordinate(1.0, 2.0, 3.0),
                'features': [0.1, 0.2, 0.3]
            }
        )
        
        result = await self.processor.process_xr_data(message)
        
        assert 'spatial_result' in result
        assert 'features' in result
        assert 'confidence' in result
        assert isinstance(result['spatial_result'], list)
        assert isinstance(result['confidence'], float)
    
    def test_input_encoder(self):
        """Test input encoding."""
        xr_data = {
            'coordinates': XRCoordinate(1.0, 2.0, 3.0),
            'features': [0.1, 0.2, 0.3]
        }
        
        encoded = self.processor.input_encoder(xr_data)
        
        assert encoded.shape[0] == 64  # Input dimensions
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_output_decoder(self):
        """Test output decoding."""
        import torch
        neural_output = torch.randn(32)
        
        decoded = self.processor.output_decoder(neural_output)
        
        assert 'spatial_result' in decoded
        assert 'features' in decoded
        assert 'confidence' in decoded


class TestSpatialAnchorAgent:
    """Test spatial anchor agent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent = SpatialAnchorAgent("test_anchor", XRCoordinate(0, 0, 0))
    
    @pytest.mark.asyncio
    async def test_process_local_data(self):
        """Test local data processing."""
        data = {
            'new_anchor': {
                'id': 'test_anchor_1',
                'position': {'x': 1.0, 'y': 2.0, 'z': 3.0}
            }
        }
        
        result = await self.agent.process_local_data(data)
        
        assert result['action'] == 'anchor_registered'
        assert 'anchor_id' in result
        assert 'position' in result
    
    @pytest.mark.asyncio
    async def test_handle_mesh_message(self):
        """Test mesh message handling."""
        message = XRMessage(
            sender_id="test_sender",
            data_type=XRDataType.SPATIAL_COORDINATES,
            payload={
                'anchor_query': True,
                'position': {'x': 0.5, 'y': 0.5, 'z': 0.5},
                'query_id': 'test_query'
            }
        )
        
        # First add an anchor
        await self.agent.process_local_data({
            'new_anchor': {
                'id': 'nearby_anchor',
                'position': {'x': 1.0, 'y': 1.0, 'z': 1.0}
            }
        })
        
        response = await self.agent.handle_mesh_message(message)
        
        if response:  # May be None if no nearby anchors
            assert response.data_type == XRDataType.SPATIAL_COORDINATES
            assert 'anchor_response' in response.payload


class TestObjectTrackerAgent:
    """Test object tracker agent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent = ObjectTrackerAgent("test_tracker", XRCoordinate(0, 0, 2))
    
    @pytest.mark.asyncio
    async def test_process_detected_objects(self):
        """Test object detection processing."""
        data = {
            'detected_objects': [
                {
                    'id': 'obj_1',
                    'position': {'x': 1.0, 'y': 1.0, 'z': 0.5},
                    'confidence': 0.9,
                    'class': 'table'
                },
                {
                    'id': 'obj_2',
                    'position': {'x': 2.0, 'y': 2.0, 'z': 1.0},
                    'confidence': 0.8,
                    'class': 'chair'
                }
            ]
        }
        
        result = await self.agent.process_local_data(data)
        
        assert result['action'] == 'objects_updated'
        assert result['updated_objects'] == ['obj_1', 'obj_2']
        assert result['total_tracked'] == 2
    
    @pytest.mark.asyncio
    async def test_low_confidence_objects_filtered(self):
        """Test that low confidence objects are filtered out."""
        data = {
            'detected_objects': [
                {
                    'id': 'obj_1',
                    'position': {'x': 1.0, 'y': 1.0, 'z': 0.5},
                    'confidence': 0.5,  # Below threshold
                    'class': 'table'
                }
            ]
        }
        
        result = await self.agent.process_local_data(data)
        
        assert result['total_tracked'] == 0


class TestXRAgentMesh:
    """Test XR agent mesh networking."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mesh = XRAgentMesh("test_mesh")
        self.agent1 = SpatialAnchorAgent("agent1", XRCoordinate(0, 0, 0))
        self.agent2 = ObjectTrackerAgent("agent2", XRCoordinate(5, 5, 1))
    
    def test_add_remove_agents(self):
        """Test adding and removing agents."""
        # Add agents
        self.mesh.add_agent(self.agent1)
        self.mesh.add_agent(self.agent2)
        
        assert len(self.mesh.agents) == 2
        assert "agent1" in self.mesh.agents
        assert "agent2" in self.mesh.agents
        
        # Remove agent
        self.mesh.remove_agent("agent1")
        
        assert len(self.mesh.agents) == 1
        assert "agent1" not in self.mesh.agents
    
    def test_connect_agents(self):
        """Test agent connections."""
        self.mesh.add_agent(self.agent1)
        self.mesh.add_agent(self.agent2)
        
        self.mesh.connect_agents("agent1", "agent2")
        
        assert "agent2" in self.mesh.topology["agent1"]
        assert "agent1" in self.mesh.topology["agent2"]
        assert self.agent2 in self.agent1.neighbors.values()
        assert self.agent1 in self.agent2.neighbors.values()
    
    def test_auto_connect_by_proximity(self):
        """Test automatic connection by proximity."""
        # Add close agents
        agent3 = SpatialAnchorAgent("agent3", XRCoordinate(1, 1, 0))  # Close to agent1
        
        self.mesh.add_agent(self.agent1)
        self.mesh.add_agent(self.agent2)
        self.mesh.add_agent(agent3)
        
        self.mesh.auto_connect_by_proximity(max_distance=3.0)
        
        # agent1 and agent3 should be connected (distance ~1.4)
        assert "agent3" in self.mesh.topology["agent1"]
        assert "agent1" in self.mesh.topology["agent3"]
        
        # agent1 and agent2 should not be connected (distance ~7.1)
        assert "agent2" not in self.mesh.topology["agent1"]
    
    @pytest.mark.asyncio
    async def test_mesh_startup_shutdown(self):
        """Test mesh startup and shutdown."""
        self.mesh.add_agent(self.agent1)
        self.mesh.add_agent(self.agent2)
        
        # Start mesh
        await self.mesh.start_mesh()
        assert self.mesh.is_running
        assert self.agent1.is_active
        assert self.agent2.is_active
        
        # Stop mesh
        await self.mesh.stop_mesh()
        assert not self.mesh.is_running
        assert not self.agent1.is_active
        assert not self.agent2.is_active
    
    @pytest.mark.asyncio
    async def test_message_broadcasting(self):
        """Test message broadcasting."""
        self.mesh.add_agent(self.agent1)
        self.mesh.add_agent(self.agent2)
        await self.mesh.start_mesh()
        
        message = XRMessage(
            sender_id="external",
            data_type=XRDataType.SPATIAL_COORDINATES,
            payload={"test": "broadcast"}
        )
        
        # Mock the receive_message method
        self.agent1.receive_message = AsyncMock()
        self.agent2.receive_message = AsyncMock()
        
        await self.mesh.broadcast_message(message)
        
        # Both agents should receive the message
        self.agent1.receive_message.assert_called_once()
        self.agent2.receive_message.assert_called_once()
        
        await self.mesh.stop_mesh()


class TestSpatialObject:
    """Test spatial object functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.obj = SpatialObject(
            object_id="test_obj",
            position=XRCoordinate(1, 2, 3),
            bounding_box={'width': 2.0, 'height': 1.0, 'depth': 1.5},
            object_class="table",
            confidence=0.9,
            semantic_features=[0.8, 0.6, 0.7]
        )
    
    def test_center_point(self):
        """Test center point calculation."""
        center = self.obj.get_center_point()
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(center, expected)
    
    def test_bounding_box_corners(self):
        """Test bounding box corner calculation."""
        corners = self.obj.get_corners()
        assert corners.shape == (8, 3)
        
        # Check that corners are around the center
        center = self.obj.get_center_point()
        distances = np.linalg.norm(corners - center, axis=1)
        
        # All corners should be within reasonable distance
        max_distance = np.sqrt(2**2 + 1**2 + 1.5**2) / 2  # Half diagonal
        assert np.all(distances <= max_distance + 1e-6)
    
    def test_overlap_detection(self):
        """Test object overlap detection."""
        # Create overlapping object
        overlapping_obj = SpatialObject(
            object_id="overlap_obj",
            position=XRCoordinate(1.5, 2.5, 3.5),  # Partially overlapping
            bounding_box={'width': 1.0, 'height': 1.0, 'depth': 1.0},
            object_class="box",
            confidence=0.8,
            semantic_features=[0.5, 0.5, 0.5]
        )
        
        # Create non-overlapping object
        distant_obj = SpatialObject(
            object_id="distant_obj",
            position=XRCoordinate(10, 10, 10),
            bounding_box={'width': 1.0, 'height': 1.0, 'depth': 1.0},
            object_class="box",
            confidence=0.8,
            semantic_features=[0.5, 0.5, 0.5]
        )
        
        assert self.obj.overlaps_with(overlapping_obj)
        assert not self.obj.overlaps_with(distant_obj)


class TestSpatialMemoryManager:
    """Test spatial memory management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.memory = SpatialMemoryManager(memory_capacity=100)
        self.test_object = SpatialObject(
            object_id="test_obj",
            position=XRCoordinate(0, 0, 0),
            bounding_box={'width': 1.0, 'height': 1.0, 'depth': 1.0},
            object_class="test",
            confidence=0.9,
            semantic_features=[0.5, 0.5, 0.5]
        )
    
    def test_add_object(self):
        """Test adding objects to memory."""
        self.memory.add_object(self.test_object)
        
        assert len(self.memory.objects) == 1
        assert "test_obj" in self.memory.objects
    
    def test_query_objects_in_radius(self):
        """Test spatial radius queries."""
        # Add objects at different positions
        obj1 = SpatialObject("obj1", XRCoordinate(1, 0, 0), {'width': 1, 'height': 1, 'depth': 1},
                           "test", 0.9, [0.5])
        obj2 = SpatialObject("obj2", XRCoordinate(10, 0, 0), {'width': 1, 'height': 1, 'depth': 1},
                           "test", 0.9, [0.5])
        
        self.memory.add_object(obj1)
        self.memory.add_object(obj2)
        
        # Query near origin
        nearby = self.memory.query_objects_in_radius(XRCoordinate(0, 0, 0), 5.0)
        
        assert len(nearby) == 1
        assert nearby[0].object_id == "obj1"
    
    def test_query_objects_by_class(self):
        """Test querying objects by class."""
        obj1 = SpatialObject("obj1", XRCoordinate(0, 0, 0), {'width': 1, 'height': 1, 'depth': 1},
                           "table", 0.9, [0.5])
        obj2 = SpatialObject("obj2", XRCoordinate(1, 0, 0), {'width': 1, 'height': 1, 'depth': 1},
                           "chair", 0.9, [0.5])
        obj3 = SpatialObject("obj3", XRCoordinate(2, 0, 0), {'width': 1, 'height': 1, 'depth': 1},
                           "table", 0.9, [0.5])
        
        self.memory.add_object(obj1)
        self.memory.add_object(obj2)
        self.memory.add_object(obj3)
        
        tables = self.memory.query_objects_by_class("table")
        
        assert len(tables) == 2
        assert all(obj.object_class == "table" for obj in tables)
    
    def test_memory_capacity_limit(self):
        """Test memory capacity enforcement."""
        small_memory = SpatialMemoryManager(memory_capacity=5)
        
        # Add more objects than capacity
        for i in range(10):
            obj = SpatialObject(f"obj{i}", XRCoordinate(i, 0, 0), 
                              {'width': 1, 'height': 1, 'depth': 1},
                              "test", 0.5, [0.5])  # Low persistence score
            small_memory.add_object(obj)
        
        # Should not exceed capacity significantly
        assert len(small_memory.objects) <= 6  # Some buffer allowed
    
    def test_persistence_updates(self):
        """Test object persistence updates."""
        self.memory.add_object(self.test_object)
        
        # Increase persistence by seeing object
        self.memory.update_object_persistence("test_obj", seen=True)
        obj = self.memory.objects["test_obj"]
        assert obj.persistence_score > 0.9
        
        # Decrease persistence by not seeing object
        for _ in range(20):  # Multiple decreases
            self.memory.update_object_persistence("test_obj", seen=False)
        
        # Object should be removed due to low persistence
        assert "test_obj" not in self.memory.objects


class TestPhotonicInteractionProcessor:
    """Test photonic interaction processor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = PhotonicInteractionProcessor()
        self.test_interaction = XRInteraction(
            interaction_id="test_interaction",
            interaction_type=InteractionType.GESTURE,
            position=XRCoordinate(1, 1, 1),
            gesture_data={
                'velocity': [1.0, 0.0, 0.0],
                'acceleration': [0.5, 0.0, 0.0],
                'direction': [1.0, 0.0, 0.0]
            },
            intensity=0.8
        )
    
    @pytest.mark.asyncio
    async def test_gesture_recognition(self):
        """Test gesture recognition processing."""
        result = await self.processor.process_gesture_recognition(self.test_interaction)
        
        assert 'gesture_type' in result
        assert 'confidence' in result
        assert 'parameters' in result
        assert isinstance(result['confidence'], float)
        assert 0.0 <= result['confidence'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_haptic_generation(self):
        """Test haptic feedback generation."""
        context_objects = [
            SpatialObject("table", XRCoordinate(1, 1, 0.8), 
                         {'width': 2, 'height': 0.8, 'depth': 1},
                         "table", 0.9, [0.8, 0.2, 0.9])
        ]
        
        haptic_feedback = await self.processor.process_haptic_generation(
            self.test_interaction, context_objects
        )
        
        assert isinstance(haptic_feedback, HapticFeedback)
        assert hasattr(haptic_feedback, 'force_vector')
        assert hasattr(haptic_feedback, 'vibration_frequency')
        assert len(haptic_feedback.force_vector) == 3
    
    @pytest.mark.asyncio
    async def test_user_intent_prediction(self):
        """Test user intent prediction."""
        interaction_history = [self.test_interaction]
        
        intent_result = await self.processor.predict_user_intent(interaction_history)
        
        assert 'predicted_intent' in intent_result
        assert 'confidence' in intent_result
        assert 'parameters' in intent_result
        assert 'next_likely_actions' in intent_result


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_successful_calls(self):
        """Test successful function calls."""
        async def success_func():
            return "success"
        
        result = await self.circuit_breaker.call(success_func)
        assert result == "success"
        assert self.circuit_breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opening(self):
        """Test circuit breaker opening on failures."""
        async def failing_func():
            raise Exception("Test failure")
        
        # Cause enough failures to open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await self.circuit_breaker.call(failing_func)
        
        assert self.circuit_breaker.state == "open"
        
        # Next call should be blocked
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await self.circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        async def failing_func():
            raise Exception("Test failure")
        
        async def success_func():
            return "success"
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await self.circuit_breaker.call(failing_func)
        
        assert self.circuit_breaker.state == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should enter half-open state and eventually close on success
        result = await self.circuit_breaker.call(success_func)
        assert result == "success"
        assert self.circuit_breaker.state == "closed"


class TestHealthMonitor:
    """Test health monitoring system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.health_monitor = HealthMonitor(check_interval=0.1)
    
    def test_component_registration(self):
        """Test component registration."""
        self.health_monitor.register_component("test_component", "xr_agent")
        
        assert "test_component" in self.health_monitor.health_metrics
        health = self.health_monitor.health_metrics["test_component"]
        assert health.status == HealthStatus.HEALTHY
    
    def test_health_metric_updates(self):
        """Test health metric updates."""
        self.health_monitor.register_component("test_component")
        
        # Update with good metrics
        self.health_monitor.update_component_health("test_component", {
            'error_rate': 0.01,
            'performance_score': 0.9,
            'resource_utilization': {'memory': 0.3, 'cpu': 0.2}
        })
        
        health = self.health_monitor.health_metrics["test_component"]
        assert health.status == HealthStatus.HEALTHY
        assert health.error_rate == 0.01
        assert health.performance_score == 0.9
    
    def test_failure_recording(self):
        """Test failure event recording."""
        self.health_monitor.register_component("test_component")
        
        failure_id = self.health_monitor.record_failure(
            "test_component", FailureType.PROCESSING_ERROR, 5,
            {"error": "test error"}
        )
        
        assert len(self.health_monitor.failure_history) == 1
        failure = self.health_monitor.failure_history[0]
        assert failure.failure_id == failure_id
        assert failure.failure_type == FailureType.PROCESSING_ERROR
        assert failure.severity == 5
        assert not failure.resolved
    
    def test_failure_resolution(self):
        """Test failure resolution."""
        self.health_monitor.register_component("test_component")
        
        failure_id = self.health_monitor.record_failure(
            "test_component", FailureType.PROCESSING_ERROR, 5
        )
        
        success = self.health_monitor.resolve_failure(failure_id, ["restart", "reset"])
        
        assert success
        failure = next(f for f in self.health_monitor.failure_history if f.failure_id == failure_id)
        assert failure.resolved
        assert failure.recovery_actions == ["restart", "reset"]
    
    def test_system_health_summary(self):
        """Test system health summary."""
        # Register multiple components with different statuses
        self.health_monitor.register_component("healthy_comp")
        self.health_monitor.register_component("degraded_comp")
        
        # Update with different health levels
        self.health_monitor.update_component_health("healthy_comp", {
            'error_rate': 0.01,
            'performance_score': 0.9
        })
        
        self.health_monitor.update_component_health("degraded_comp", {
            'error_rate': 0.15,
            'performance_score': 0.5
        })
        
        summary = self.health_monitor.get_system_health_summary()
        
        assert summary['components'] == 2
        assert summary['healthy_components'] >= 1
        assert summary['status'] in ['healthy', 'degraded', 'critical']


class TestReliableXRAgent:
    """Test reliable XR agent functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.health_monitor = HealthMonitor()
        self.agent = ReliableXRAgent(
            "reliable_agent", XRAgentType.SPATIAL_ANCHOR, XRCoordinate(0, 0, 0)
        )
        self.agent.set_health_monitor(self.health_monitor)
    
    @pytest.mark.asyncio
    async def test_reliable_messaging(self):
        """Test reliable message sending."""
        message = XRMessage(
            sender_id="test",
            data_type=XRDataType.SPATIAL_COORDINATES,
            payload={"test": "data"}
        )
        
        # Mock the underlying send_message
        self.agent.send_message = AsyncMock()
        
        await self.agent.reliable_send_message(message)
        
        self.agent.send_message.assert_called_once_with(message)
    
    def test_reliability_summary(self):
        """Test reliability metrics summary."""
        summary = self.agent.get_reliability_summary()
        
        assert 'agent_id' in summary
        assert 'circuit_breaker_state' in summary
        assert 'reliability_metrics' in summary
        assert 'performance_metrics' in summary
        assert summary['agent_id'] == "reliable_agent"


class TestIntegration:
    """Integration tests for complete XR system."""
    
    @pytest.mark.asyncio
    async def test_complete_xr_workflow(self):
        """Test complete XR workflow integration."""
        # Create XR mesh with reliable agents
        mesh = XRAgentMesh("integration_test")
        health_monitor = HealthMonitor()
        
        # Create agents
        anchor_agent = ReliableXRAgent(
            "anchor", XRAgentType.SPATIAL_ANCHOR, XRCoordinate(0, 0, 0)
        )
        tracker_agent = ReliableXRAgent(
            "tracker", XRAgentType.OBJECT_TRACKER, XRCoordinate(5, 5, 1)
        )
        
        anchor_agent.set_health_monitor(health_monitor)
        tracker_agent.set_health_monitor(health_monitor)
        
        # Add to mesh
        mesh.add_agent(anchor_agent)
        mesh.add_agent(tracker_agent)
        mesh.auto_connect_by_proximity(max_distance=10.0)
        
        # Start monitoring and mesh
        await health_monitor.start_monitoring()
        await mesh.start_mesh()
        
        try:
            # Simulate some activity
            await asyncio.sleep(0.5)
            
            # Check that everything is running
            assert mesh.is_running
            assert anchor_agent.is_active
            assert tracker_agent.is_active
            
            # Check health monitoring
            health_summary = health_monitor.get_system_health_summary()
            assert health_summary['components'] == 2
            
            # Check mesh status
            mesh_status = mesh.get_mesh_status()
            assert mesh_status['is_running']
            assert mesh_status['active_agents'] == 2
            
        finally:
            await mesh.stop_mesh()
            health_monitor.stop_monitoring()
    
    @pytest.mark.asyncio  
    async def test_spatial_computing_integration(self):
        """Test spatial computing with memory management."""
        processor = PhotonicSpatialProcessor()
        memory_manager = SpatialMemoryManager()
        
        # Simulate sensor data
        sensor_data = {
            'depth': np.random.rand(32, 32),
            'rgb': np.random.rand(32, 32, 3),
            'point_cloud': np.random.rand(100, 3)
        }
        
        # Process object detection
        detected_objects = await processor.process_object_detection(sensor_data)
        
        # Add to memory
        for obj in detected_objects:
            memory_manager.add_object(obj)
        
        # Query memory
        if detected_objects:
            center = detected_objects[0].position
            nearby = memory_manager.query_objects_in_radius(center, 5.0)
            assert len(nearby) >= 1
        
        # Check memory stats
        stats = memory_manager.get_memory_stats()
        assert stats['total_objects'] == len(detected_objects)


# Performance benchmarks
class TestPerformance:
    """Performance tests for XR system."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_message_processing_throughput(self):
        """Benchmark message processing throughput."""
        processor = PhotonicXRProcessor(input_dimensions=128, output_dimensions=64)
        
        # Create test messages
        messages = []
        for i in range(100):
            message = XRMessage(
                sender_id=f"sender_{i}",
                data_type=XRDataType.SPATIAL_COORDINATES,
                payload={
                    'coordinates': XRCoordinate(
                        np.random.uniform(-10, 10),
                        np.random.uniform(-10, 10),
                        np.random.uniform(0, 5)
                    ),
                    'features': np.random.randn(100).tolist()
                }
            )
            messages.append(message)
        
        # Measure processing time
        start_time = time.time()
        
        for message in messages:
            await processor.process_xr_data(message)
        
        total_time = time.time() - start_time
        throughput = len(messages) / total_time
        
        print(f"Message processing throughput: {throughput:.1f} messages/second")
        assert throughput > 10  # Should process at least 10 messages/second
    
    @pytest.mark.benchmark
    def test_spatial_query_performance(self):
        """Benchmark spatial query performance."""
        memory_manager = SpatialMemoryManager(memory_capacity=1000)
        
        # Add many objects
        for i in range(1000):
            obj = SpatialObject(
                object_id=f"obj_{i}",
                position=XRCoordinate(
                    np.random.uniform(-50, 50),
                    np.random.uniform(-50, 50),
                    np.random.uniform(0, 10)
                ),
                bounding_box={'width': 1, 'height': 1, 'depth': 1},
                object_class=f"class_{i % 10}",
                confidence=0.8,
                semantic_features=[0.5] * 5
            )
            memory_manager.add_object(obj)
        
        # Measure query time
        query_center = XRCoordinate(0, 0, 5)
        
        start_time = time.time()
        
        for _ in range(100):
            nearby = memory_manager.query_objects_in_radius(query_center, 10.0)
        
        total_time = time.time() - start_time
        query_rate = 100 / total_time
        
        print(f"Spatial query rate: {query_rate:.1f} queries/second")
        assert query_rate > 50  # Should handle at least 50 queries/second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])