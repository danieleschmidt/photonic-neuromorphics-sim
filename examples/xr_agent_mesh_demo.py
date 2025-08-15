#!/usr/bin/env python3
"""
XR Agent Mesh Demonstration for Photonic Neuromorphic Systems.

This example demonstrates the complete XR agent mesh system with:
- Distributed photonic neural processing
- Real-time spatial computing
- Interactive visualization
- Ultra-low latency communication

Usage:
    python examples/xr_agent_mesh_demo.py
"""

import asyncio
import sys
import os
import json
from datetime import datetime
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from photonic_neuromorphics import (
    # XR Agent Mesh
    XRAgentMesh, XRAgent, SpatialAnchorAgent, ObjectTrackerAgent,
    PhotonicXRProcessor, XRCoordinate, XRMessage, XRDataType, XRAgentType,
    
    # XR Spatial Computing
    PhotonicSpatialProcessor, SpatialMemoryManager, SpatialObject, SpatialRegion,
    
    # XR Visualization
    PhotonicInteractionProcessor, XRVisualizationEngine, XRInteraction, HapticFeedback,
    InteractionType, RenderingMode,
    
    # Monitoring
    MetricsCollector
)


class EnvironmentMapperAgent(XRAgent):
    """Agent specialized for environment mapping and SLAM."""
    
    def __init__(self, agent_id: str, position: XRCoordinate):
        super().__init__(agent_id, XRAgentType.ENVIRONMENT_MAPPER, position)
        self.spatial_processor = PhotonicSpatialProcessor()
        self.memory_manager = SpatialMemoryManager()
        self.mapped_regions: Dict[str, SpatialRegion] = {}
        
    async def process_local_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment mapping data."""
        if 'sensor_data' in data:
            # Detect objects in environment
            detected_objects = await self.spatial_processor.process_object_detection(
                data['sensor_data']
            )
            
            # Add to spatial memory
            for obj in detected_objects:
                self.memory_manager.add_object(obj)
            
            # Analyze spatial relationships
            relationships = await self.spatial_processor.process_spatial_relationships(
                detected_objects
            )
            
            return {
                'action': 'environment_mapped',
                'objects_detected': len(detected_objects),
                'relationships_found': len(relationships['relationships']),
                'memory_usage': self.memory_manager.get_memory_stats()
            }
        
        return {'action': 'no_action'}
    
    async def handle_mesh_message(self, message: XRMessage) -> Optional[XRMessage]:
        """Handle environment mapping mesh messages."""
        if message.data_type == XRDataType.ENVIRONMENT_MAP:
            payload = message.payload
            
            if 'map_query' in payload:
                # Return spatial map for requested region
                query_center = XRCoordinate(**payload['center'])
                radius = payload.get('radius', 5.0)
                
                nearby_objects = self.memory_manager.query_objects_in_radius(
                    query_center, radius
                )
                
                return XRMessage(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    data_type=XRDataType.ENVIRONMENT_MAP,
                    payload={
                        'map_response': [
                            {
                                'object_id': obj.object_id,
                                'position': {
                                    'x': obj.position.x,
                                    'y': obj.position.y,
                                    'z': obj.position.z
                                },
                                'object_class': obj.object_class,
                                'confidence': obj.confidence
                            }
                            for obj in nearby_objects
                        ],
                        'query_id': payload.get('query_id')
                    }
                )
        
        return None


class CollaborationManagerAgent(XRAgent):
    """Agent for managing multi-user XR collaboration."""
    
    def __init__(self, agent_id: str, position: XRCoordinate):
        super().__init__(agent_id, XRAgentType.COLLABORATION_MANAGER, position)
        self.active_users: Dict[str, Dict[str, Any]] = {}
        self.shared_annotations: List[Dict[str, Any]] = []
        self.collaboration_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def process_local_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process collaboration data."""
        if 'user_action' in data:
            user_id = data['user_id']
            action = data['user_action']
            
            # Update user state
            self.active_users[user_id] = {
                'position': data.get('position', XRCoordinate(0, 0, 0)),
                'last_action': action,
                'timestamp': datetime.now().timestamp()
            }
            
            if action['type'] == 'annotate':
                # Add shared annotation
                annotation = {
                    'id': f"annotation_{len(self.shared_annotations)}",
                    'user_id': user_id,
                    'position': action['position'],
                    'content': action['content'],
                    'timestamp': datetime.now().timestamp()
                }
                self.shared_annotations.append(annotation)
                
                return {
                    'action': 'annotation_shared',
                    'annotation_id': annotation['id'],
                    'active_users': len(self.active_users)
                }
            
        return {'action': 'collaboration_updated'}
    
    async def handle_mesh_message(self, message: XRMessage) -> Optional[XRMessage]:
        """Handle collaboration mesh messages."""
        if message.data_type == XRDataType.SPATIAL_COORDINATES:
            payload = message.payload
            
            if 'collaboration_request' in payload:
                # Handle collaboration request
                session_id = f"session_{datetime.now().timestamp():.6f}"
                
                self.collaboration_sessions[session_id] = {
                    'participants': [message.sender_id],
                    'created_at': datetime.now().timestamp(),
                    'shared_objects': [],
                    'annotations': []
                }
                
                return XRMessage(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    data_type=XRDataType.SPATIAL_COORDINATES,
                    payload={
                        'collaboration_response': {
                            'session_id': session_id,
                            'status': 'created'
                        }
                    }
                )
        
        return None


async def create_comprehensive_xr_demo():
    """Create a comprehensive XR demonstration environment."""
    print("🚀 Creating Comprehensive XR Agent Mesh Demo")
    print("=" * 60)
    
    # Initialize metrics collection
    metrics_collector = MetricsCollector()
    
    # Create XR mesh
    mesh = XRAgentMesh("comprehensive_xr_demo")
    mesh.set_metrics_collector(metrics_collector)
    
    # Create diverse agents
    agents = [
        # Spatial anchors
        SpatialAnchorAgent("anchor_origin", XRCoordinate(0, 0, 0)),
        SpatialAnchorAgent("anchor_northeast", XRCoordinate(10, 10, 0)),
        SpatialAnchorAgent("anchor_northwest", XRCoordinate(-10, 10, 0)),
        
        # Object trackers
        ObjectTrackerAgent("tracker_ceiling", XRCoordinate(0, 0, 3)),
        ObjectTrackerAgent("tracker_ground", XRCoordinate(0, 0, 0.5)),
        
        # Environment mapper
        EnvironmentMapperAgent("mapper_main", XRCoordinate(5, 5, 1.5)),
        
        # Collaboration manager
        CollaborationManagerAgent("collab_mgr", XRCoordinate(0, 0, 2))
    ]
    
    # Add agents to mesh
    for agent in agents:
        mesh.add_agent(agent)
    
    # Create proximity-based connections
    mesh.auto_connect_by_proximity(max_distance=12.0)
    
    # Create additional strategic connections
    mesh.connect_agents("mapper_main", "collab_mgr")
    mesh.connect_agents("tracker_ceiling", "tracker_ground")
    
    print(f"✅ Created mesh with {len(agents)} agents")
    print(f"📊 Topology: {mesh.topology}")
    
    return mesh, agents, metrics_collector


async def simulate_xr_interactions(mesh: XRAgentMesh, agents: List[XRAgent], 
                                 duration: float = 15.0):
    """Simulate realistic XR interactions."""
    print(f"\n🎯 Simulating XR Interactions for {duration} seconds")
    print("-" * 40)
    
    # Initialize interaction processor
    interaction_processor = PhotonicInteractionProcessor()
    interaction_processor.set_metrics_collector(mesh._metrics_collector)
    
    # Initialize visualization engine
    viz_engine = XRVisualizationEngine()
    viz_engine.set_metrics_collector(mesh._metrics_collector)
    
    # Create demo spatial objects
    demo_objects = [
        SpatialObject(
            object_id="conference_table",
            position=XRCoordinate(2, 2, 0.8),
            bounding_box={'width': 3.0, 'height': 0.8, 'depth': 1.5},
            object_class="table",
            confidence=0.95,
            semantic_features=[0.9, 0.8, 0.7]
        ),
        SpatialObject(
            object_id="whiteboard",
            position=XRCoordinate(-3, 5, 1.5),
            bounding_box={'width': 0.1, 'height': 1.2, 'depth': 2.0},
            object_class="whiteboard",
            confidence=0.92,
            semantic_features=[0.8, 0.9, 0.6]
        ),
        SpatialObject(
            object_id="holographic_display",
            position=XRCoordinate(0, 0, 2.5),
            bounding_box={'width': 1.0, 'height': 1.0, 'depth': 0.1},
            object_class="display",
            confidence=0.88,
            semantic_features=[0.7, 0.6, 0.9]
        )
    ]
    
    # Add objects to visualization
    for obj in demo_objects:
        viz_engine.add_spatial_object(obj)
    
    # Start visualization
    viz_engine.start_rendering()
    
    # Start the mesh
    await mesh.start_mesh()
    
    try:
        start_time = datetime.now().timestamp()
        interaction_count = 0
        
        while datetime.now().timestamp() - start_time < duration:
            current_time = datetime.now().timestamp() - start_time
            
            # Generate interaction based on scenario progression
            if current_time < 5.0:
                # Initial exploration phase
                interaction = XRInteraction(
                    interaction_id=f"explore_{interaction_count}",
                    interaction_type=InteractionType.GAZE,
                    position=XRCoordinate(
                        np.random.uniform(-5, 5),
                        np.random.uniform(-5, 5),
                        np.random.uniform(0.5, 2.5)
                    ),
                    intensity=0.6
                )
            elif current_time < 10.0:
                # Active manipulation phase
                interaction = XRInteraction(
                    interaction_id=f"manipulate_{interaction_count}",
                    interaction_type=InteractionType.GESTURE,
                    position=XRCoordinate(
                        np.random.uniform(-2, 4),
                        np.random.uniform(0, 4),
                        np.random.uniform(0.8, 1.8)
                    ),
                    gesture_data={
                        'velocity': [np.random.uniform(-2, 2) for _ in range(3)],
                        'acceleration': [np.random.uniform(-1, 1) for _ in range(3)],
                        'direction': [1.0, 0.0, 0.0]
                    },
                    intensity=0.9,
                    duration=0.5
                )
            else:
                # Collaboration phase
                interaction = XRInteraction(
                    interaction_id=f"collaborate_{interaction_count}",
                    interaction_type=InteractionType.VOICE,
                    position=XRCoordinate(0, 0, 1.7),
                    intensity=0.8,
                    duration=1.0
                )
            
            # Process interaction through photonic neural networks
            gesture_result = await interaction_processor.process_gesture_recognition(interaction)
            haptic_feedback = await interaction_processor.process_haptic_generation(
                interaction, demo_objects
            )
            
            # Add to visualization
            viz_engine.add_interaction_point(interaction)
            viz_engine.add_haptic_feedback(haptic_feedback)
            
            # Send environment mapping request every 2 seconds
            if interaction_count % 20 == 0:
                map_query = XRMessage(
                    sender_id="demo_client",
                    data_type=XRDataType.ENVIRONMENT_MAP,
                    payload={
                        'map_query': True,
                        'center': {'x': 0, 'y': 0, 'z': 1},
                        'radius': 8.0,
                        'query_id': f"map_query_{interaction_count}"
                    }
                )
                await mesh.broadcast_message(map_query)
            
            # Send collaboration request every 5 seconds
            if interaction_count % 50 == 0:
                collab_request = XRMessage(
                    sender_id="demo_user",
                    data_type=XRDataType.SPATIAL_COORDINATES,
                    payload={
                        'collaboration_request': {
                            'user_id': 'demo_user',
                            'session_type': 'design_review'
                        }
                    }
                )
                await mesh.broadcast_message(collab_request)
            
            interaction_count += 1
            await asyncio.sleep(0.1)  # 10Hz interaction rate
        
        # Get final performance metrics
        mesh_status = mesh.get_mesh_status()
        interaction_performance = interaction_processor.get_performance_summary()
        rendering_stats = viz_engine.get_rendering_stats()
        
        print(f"\n📈 Simulation Complete!")
        print(f"   Total interactions: {interaction_count}")
        print(f"   Active agents: {mesh_status['active_agents']}")
        print(f"   Network utilization: {mesh_status['mesh_metrics']['network_utilization']:.1%}")
        print(f"   Average gesture processing: {interaction_performance.get('gesture_recognition', {}).get('avg_time', 0)*1e6:.1f} μs")
        print(f"   Average haptic processing: {interaction_performance.get('haptic_generation', {}).get('avg_time', 0)*1e6:.1f} μs")
        print(f"   Rendering FPS: {rendering_stats['avg_fps']:.1f}")
        
        return {
            'interaction_count': interaction_count,
            'mesh_status': mesh_status,
            'interaction_performance': interaction_performance,
            'rendering_stats': rendering_stats,
            'demo_objects': len(demo_objects)
        }
        
    finally:
        await mesh.stop_mesh()
        viz_engine.stop_rendering()


async def demonstrate_photonic_advantages():
    """Demonstrate the advantages of photonic neural processing for XR."""
    print(f"\n⚡ Demonstrating Photonic Neural Processing Advantages")
    print("-" * 50)
    
    # Create comparison processors
    photonic_processor = PhotonicXRProcessor(
        input_dimensions=256,
        output_dimensions=128,
        processing_layers=[256, 512, 256, 128],
        wavelength=1550e-9
    )
    
    # Simulate processing workload
    test_messages = []
    for i in range(100):
        message = XRMessage(
            sender_id=f"test_sender_{i}",
            data_type=XRDataType.SPATIAL_COORDINATES,
            payload={
                'coordinates': XRCoordinate(
                    np.random.uniform(-10, 10),
                    np.random.uniform(-10, 10),
                    np.random.uniform(0, 5)
                ),
                'features': np.random.randn(200).tolist(),
                'timestamp': datetime.now().timestamp()
            }
        )
        test_messages.append(message)
    
    # Process messages and measure performance
    processing_times = []
    energy_consumption = []
    
    print("Processing 100 XR messages through photonic neural networks...")
    
    for i, message in enumerate(test_messages):
        start_time = datetime.now().timestamp()
        
        result = await photonic_processor.process_xr_data(message)
        
        processing_time = datetime.now().timestamp() - start_time
        processing_times.append(processing_time)
        energy_consumption.append(photonic_processor.energy_consumption)
        
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/100 messages")
    
    # Calculate statistics
    avg_processing_time = np.mean(processing_times)
    total_energy = np.sum(energy_consumption)
    throughput = len(test_messages) / np.sum(processing_times)
    
    print(f"\n🔬 Photonic Processing Results:")
    print(f"   Average processing time: {avg_processing_time*1e6:.1f} μs")
    print(f"   Min processing time: {np.min(processing_times)*1e6:.1f} μs")
    print(f"   Max processing time: {np.max(processing_times)*1e6:.1f} μs")
    print(f"   Total energy consumption: {total_energy*1e12:.2f} pJ")
    print(f"   Energy per message: {total_energy/len(test_messages)*1e12:.2f} pJ")
    print(f"   Message throughput: {throughput:.0f} messages/second")
    
    # Compare with theoretical electronic equivalent
    electronic_energy_per_message = 50e-12  # 50 pJ (typical electronic SNN)
    electronic_processing_time = 100e-6     # 100 μs (typical electronic processing)
    
    energy_improvement = electronic_energy_per_message / (total_energy/len(test_messages))
    speed_improvement = electronic_processing_time / avg_processing_time
    
    print(f"\n🚀 Photonic Advantages:")
    print(f"   Energy efficiency improvement: {energy_improvement:.0f}×")
    print(f"   Processing speed improvement: {speed_improvement:.0f}×")
    print(f"   Suitable for real-time XR: {avg_processing_time < 1e-3}")  # < 1ms threshold
    
    return {
        'avg_processing_time': avg_processing_time,
        'total_energy': total_energy,
        'throughput': throughput,
        'energy_improvement': energy_improvement,
        'speed_improvement': speed_improvement
    }


async def main():
    """Run the comprehensive XR agent mesh demonstration."""
    print("🌟 Photonic Neuromorphic XR Agent Mesh Demonstration")
    print("=" * 60)
    print("This demo showcases:")
    print("• Distributed photonic neural processing")
    print("• Real-time XR spatial computing")
    print("• Multi-agent coordination")
    print("• Ultra-low latency interactions")
    print("• Haptic feedback generation")
    print("• Collaborative XR environments")
    print("=" * 60)
    
    try:
        # Phase 1: Create XR environment
        mesh, agents, metrics_collector = await create_comprehensive_xr_demo()
        
        # Phase 2: Simulate XR interactions
        simulation_results = await simulate_xr_interactions(mesh, agents, duration=10.0)
        
        # Phase 3: Demonstrate photonic advantages
        photonic_results = await demonstrate_photonic_advantages()
        
        # Phase 4: Generate comprehensive report
        final_metrics = metrics_collector.get_metrics_summary()
        
        print(f"\n📋 Final Demo Report")
        print("=" * 40)
        print(f"Mesh agents created: {len(agents)}")
        print(f"Interactions simulated: {simulation_results['interaction_count']}")
        print(f"Network efficiency: {simulation_results['mesh_status']['mesh_metrics']['network_utilization']:.1%}")
        print(f"Photonic energy advantage: {photonic_results['energy_improvement']:.0f}×")
        print(f"Photonic speed advantage: {photonic_results['speed_improvement']:.0f}×")
        print(f"Total metrics collected: {len(final_metrics)}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'simulation_results': simulation_results,
            'photonic_advantages': photonic_results,
            'metrics_summary': final_metrics,
            'demo_config': {
                'agents_count': len(agents),
                'simulation_duration': 10.0,
                'interaction_rate': '10Hz'
            }
        }
        
        results_file = f"xr_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        print(f"🎉 XR Agent Mesh Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    exit(exit_code)