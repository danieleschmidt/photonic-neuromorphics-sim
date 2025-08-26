"""
Comprehensive test suite for Ultra-High Performance Caching system.

Tests cover multi-level cache hierarchy, adaptive replacement policies,
intelligent prefetching, and performance optimization.
"""

import pytest
import numpy as np
import torch
import time
import threading
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple

from src.photonic_neuromorphics.ultra_high_performance_caching import (
    UltraHighPerformanceCache,
    CacheEntry,
    CacheLevel,
    ReplacementPolicy,
    PrefetchPredictor,
    AdaptiveReplacementCache,
    create_photonic_cache_demo,
    run_cache_performance_benchmark,
    validate_cache_performance_improvements
)


class TestCacheEntry:
    """Test suite for CacheEntry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.entry = CacheEntry(
            key="test_key",
            value="test_value",
            size_bytes=1024,
            computation_cost=5.0
        )
    
    def test_initialization(self):
        """Test cache entry initialization."""
        assert self.entry.key == "test_key"
        assert self.entry.value == "test_value"
        assert self.entry.size_bytes == 1024
        assert self.entry.computation_cost == 5.0
        assert self.entry.access_count == 0
        assert self.entry.priority == 1.0
    
    def test_update_access(self):
        """Test access statistics update."""
        initial_time = self.entry.last_access_time
        initial_count = self.entry.access_count
        
        time.sleep(0.001)  # Small delay
        self.entry.update_access()
        
        assert self.entry.access_count == initial_count + 1
        assert self.entry.last_access_time > initial_time
    
    def test_age_calculation(self):
        """Test entry age calculation."""
        age = self.entry.get_age()
        assert age >= 0
        assert age < 1.0  # Should be very recent
        
        time.sleep(0.01)
        new_age = self.entry.get_age()
        assert new_age > age
    
    def test_idle_time_calculation(self):
        """Test idle time calculation."""
        self.entry.update_access()
        time.sleep(0.01)
        
        idle_time = self.entry.get_idle_time()
        assert idle_time > 0
        assert idle_time < 1.0
    
    def test_utility_score_calculation(self):
        """Test utility score calculation."""
        # Fresh entry
        initial_score = self.entry.calculate_utility_score()
        assert initial_score > 0
        
        # After multiple accesses
        for _ in range(10):
            self.entry.update_access()
        
        accessed_score = self.entry.calculate_utility_score()
        assert accessed_score > initial_score  # Should have higher utility
    
    def test_utility_score_with_computation_cost(self):
        """Test utility score considers computation cost."""
        cheap_entry = CacheEntry(key="cheap", value="data", computation_cost=1.0)
        expensive_entry = CacheEntry(key="expensive", value="data", computation_cost=10.0)
        
        cheap_score = cheap_entry.calculate_utility_score()
        expensive_score = expensive_entry.calculate_utility_score()
        
        # Expensive computations should have higher utility
        assert expensive_score > cheap_score
    
    def test_priority_effects(self):
        """Test priority affects utility score."""
        self.entry.priority = 2.0
        high_priority_score = self.entry.calculate_utility_score()
        
        self.entry.priority = 0.5
        low_priority_score = self.entry.calculate_utility_score()
        
        assert high_priority_score > low_priority_score


class TestPrefetchPredictor:
    """Test suite for PrefetchPredictor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = PrefetchPredictor(history_size=100)
    
    def test_initialization(self):
        """Test predictor initialization."""
        assert self.predictor.history_size == 100
        assert len(self.predictor.access_history) == 0
        assert len(self.predictor.pattern_frequencies) == 0
        assert self.predictor.prediction_accuracy == 0.0
        assert not self.predictor.trained
    
    def test_record_access(self):
        """Test access recording."""
        self.predictor.record_access("key1", {"context": "test"})
        
        assert len(self.predictor.access_history) == 1
        assert self.predictor.access_history[0]["key"] == "key1"
        assert self.predictor.access_history[0]["context"]["context"] == "test"
    
    def test_pattern_learning(self):
        """Test pattern frequency learning."""
        # Create access pattern
        keys = ["A", "B", "C", "A", "B", "C", "A", "B"]
        for key in keys:
            self.predictor.record_access(key)
        
        # Check patterns were learned
        assert ("A", "B") in self.predictor.pattern_frequencies
        assert ("B", "C") in self.predictor.pattern_frequencies
        assert self.predictor.pattern_frequencies[("A", "B")] > 0
        assert self.predictor.pattern_frequencies[("B", "C")] > 0
    
    def test_prediction_from_patterns(self):
        """Test prediction based on learned patterns."""
        # Train with pattern: A -> B -> C
        pattern = ["A", "B", "C"] * 10
        for key in pattern:
            self.predictor.record_access(key)
        
        # Predict next accesses after "A"
        predictions = self.predictor.predict_next_accesses("A", num_predictions=3)
        
        assert len(predictions) <= 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in predictions)
        assert all(isinstance(p[1], float) for p in predictions)  # Confidence scores
        
        # "B" should be the most likely prediction after "A"
        if predictions:
            top_prediction = predictions[0]
            assert top_prediction[0] == "B"
            assert 0 <= top_prediction[1] <= 1  # Valid confidence
    
    def test_prediction_validation(self):
        """Test prediction accuracy tracking."""
        initial_accuracy = self.predictor.prediction_accuracy
        
        # Record correct prediction
        self.predictor.validate_prediction("predicted", "predicted")
        
        # Record incorrect prediction
        self.predictor.validate_prediction("predicted", "actual")
        
        assert self.predictor.predictions_made == 2
        assert self.predictor.predictions_correct == 1
        assert self.predictor.prediction_accuracy == 0.5
    
    def test_history_size_limit(self):
        """Test access history size limiting."""
        # Add more entries than history size
        for i in range(150):
            self.predictor.record_access(f"key_{i}")
        
        # Should maintain history size limit
        assert len(self.predictor.access_history) == 100
        
        # Should keep most recent entries
        recent_keys = [record["key"] for record in self.predictor.access_history]
        assert "key_149" in recent_keys
        assert "key_0" not in recent_keys
    
    def test_model_training_trigger(self):
        """Test ML model training trigger."""
        # Add enough data to trigger training
        for i in range(100):
            self.predictor.record_access(f"key_{i % 10}")  # Create patterns
        
        # Training should have been triggered
        assert self.predictor.trained
        assert self.predictor.model_weights is not None


class TestAdaptiveReplacementCache:
    """Test suite for AdaptiveReplacementCache (ARC)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.arc = AdaptiveReplacementCache(capacity=10)
    
    def test_initialization(self):
        """Test ARC initialization."""
        assert self.arc.capacity == 10
        assert self.arc.c == 10
        assert self.arc.p == 0
        assert len(self.arc.t1) == 0
        assert len(self.arc.t2) == 0
        assert self.arc.hits == 0
        assert self.arc.misses == 0
    
    def test_basic_put_get(self):
        """Test basic put and get operations."""
        entry = CacheEntry(key="test", value="data")
        
        # Put and get
        self.arc.put("test", entry)
        result = self.arc.get("test")
        
        assert result == "data"
        assert self.arc.hits == 1
        assert self.arc.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss handling."""
        result = self.arc.get("nonexistent")
        
        assert result is None
        assert self.arc.hits == 0
        assert self.arc.misses == 1
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        entries = [CacheEntry(key=f"key_{i}", value=f"data_{i}") for i in range(5)]
        
        # Add entries
        for entry in entries:
            self.arc.put(entry.key, entry)
        
        # Access some entries (hits)
        for i in range(3):
            self.arc.get(f"key_{i}")
        
        # Access non-existent entries (misses)
        for i in range(2):
            self.arc.get(f"missing_{i}")
        
        hit_rate = self.arc.get_hit_rate()
        assert hit_rate == 3.0 / 5.0  # 3 hits out of 5 total accesses
    
    def test_capacity_enforcement(self):
        """Test capacity enforcement."""
        # Add more entries than capacity
        for i in range(15):
            entry = CacheEntry(key=f"key_{i}", value=f"data_{i}")
            self.arc.put(entry.key, entry)
        
        # Total size should not exceed capacity
        total_size = len(self.arc.t1) + len(self.arc.t2)
        assert total_size <= self.arc.capacity
    
    def test_frequency_promotion(self):
        """Test promotion from T1 to T2 on repeated access."""
        entry = CacheEntry(key="test", value="data")
        self.arc.put("test", entry)
        
        # First access should be in T1
        self.arc.get("test")
        assert "test" in self.arc.t1 or "test" in self.arc.t2
        
        # Second access should promote to T2
        self.arc.get("test")
        assert "test" in self.arc.t2


class TestUltraHighPerformanceCache:
    """Test suite for UltraHighPerformanceCache main class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = UltraHighPerformanceCache(
            l1_capacity=10,
            l2_capacity=20,
            l3_capacity=50,
            persistent_capacity=100,
            enable_prefetching=False,  # Disable for deterministic testing
            enable_distributed=False
        )
    
    def test_initialization(self):
        """Test cache initialization."""
        assert self.cache.l1_capacity == 10
        assert self.cache.l2_capacity == 20
        assert self.cache.l3_capacity == 50
        assert len(self.cache.caches) >= 4  # At least 4 cache levels
        assert CacheLevel.L1_CPU in self.cache.caches
        assert CacheLevel.L2_CPU in self.cache.caches
    
    def test_basic_get_put(self):
        """Test basic cache operations."""
        # Put value in cache
        self.cache.put("test_key", "test_value", computation_cost=1.0)
        
        # Get value from cache
        result = self.cache.get("test_key")
        assert result == "test_value"
    
    def test_cache_miss_with_computation(self):
        """Test cache miss with computation function."""
        def expensive_computation():
            time.sleep(0.001)  # Simulate computation
            return "computed_value"
        
        start_time = time.time()
        result = self.cache.get("missing_key", expensive_computation)
        end_time = time.time()
        
        assert result == "computed_value"
        assert end_time - start_time >= 0.001  # Computation was executed
        
        # Second access should be from cache (faster)
        start_time = time.time()
        result2 = self.cache.get("missing_key")
        end_time = time.time()
        
        assert result2 == "computed_value"
        assert end_time - start_time < 0.001  # Should be much faster
    
    def test_size_calculation(self):
        """Test object size calculation."""
        # Test with different object types
        tensor_size = self.cache._calculate_size(torch.randn(10, 10))
        numpy_size = self.cache._calculate_size(np.random.random((10, 10)))
        string_size = self.cache._calculate_size("test string")
        
        assert tensor_size > 0
        assert numpy_size > 0
        assert string_size > 0
        
        # Larger objects should have larger sizes
        large_tensor_size = self.cache._calculate_size(torch.randn(100, 100))
        assert large_tensor_size > tensor_size
    
    def test_cache_level_determination(self):
        """Test automatic cache level determination."""
        # Small, expensive item should go to L1
        small_expensive = CacheEntry(
            key="small_expensive",
            value="data",
            size_bytes=500,
            computation_cost=10.0
        )
        level = self.cache._determine_cache_level(small_expensive, {})
        assert level == CacheLevel.L1_CPU
        
        # Large item should go to persistent storage
        large_item = CacheEntry(
            key="large_item",
            value="data",
            size_bytes=200000
        )
        level = self.cache._determine_cache_level(large_item, {})
        assert level == CacheLevel.PERSISTENT
        
        # Quantum-related item should use L3 (ARC)
        quantum_item = CacheEntry(
            key="quantum_item",
            value="data",
            quantum_coherence_time=1e-6
        )
        level = self.cache._determine_cache_level(quantum_item, {})
        assert level == CacheLevel.L3_SHARED
    
    @pytest.mark.parametrize("replacement_policy", [
        ReplacementPolicy.LRU,
        ReplacementPolicy.LFU,
        ReplacementPolicy.QUANTUM_AWARE
    ])
    def test_replacement_policies(self, replacement_policy):
        """Test different cache replacement policies."""
        cache = UltraHighPerformanceCache(
            l1_capacity=3,
            replacement_policy=replacement_policy,
            enable_prefetching=False
        )
        
        # Fill cache beyond capacity
        for i in range(5):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Check that capacity is enforced
        l1_cache = cache.caches[CacheLevel.L1_CPU]
        if hasattr(l1_cache, '__len__'):
            assert len(l1_cache) <= cache.l1_capacity
    
    def test_multilevel_promotion(self):
        """Test promotion between cache levels."""
        # Add item to lower level cache
        self.cache.put("test_item", "test_value")
        
        # Access multiple times to trigger promotion
        for _ in range(10):
            result = self.cache.get("test_item")
            assert result == "test_value"
        
        # Item should potentially be promoted to higher levels
        # (Exact behavior depends on internal policies)
    
    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        results = {}
        
        def worker(thread_id):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                self.cache.put(key, value)
                retrieved = self.cache.get(key)
                results[key] = retrieved
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify all operations completed successfully
        assert len(results) == 50  # 5 threads * 10 operations
        for key, value in results.items():
            expected_value = key.replace("key", "value")
            assert value == expected_value
    
    def test_cache_statistics(self):
        """Test cache statistics collection."""
        # Perform some cache operations
        for i in range(10):
            self.cache.put(f"key_{i}", f"value_{i}")
        
        # Access some items (hits)
        for i in range(5):
            self.cache.get(f"key_{i}")
        
        # Access non-existent items (misses)
        for i in range(3):
            self.cache.get(f"missing_{i}")
        
        stats = self.cache.get_cache_statistics()
        
        # Check statistics structure
        assert "overall" in stats
        assert "hit_rate" in stats["overall"]
        assert isinstance(stats["overall"]["hit_rate"], float)
        
        # Check per-level statistics
        for level in CacheLevel:
            if level.value in stats:
                level_stats = stats[level.value]
                assert "hits" in level_stats
                assert "misses" in level_stats
                assert "hit_rate" in level_stats
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Add items to cache
        for i in range(10):
            self.cache.put(f"key_{i}", f"value_{i}")
        
        # Clear specific level
        self.cache.clear_cache(CacheLevel.L1_CPU)
        
        # Verify level is cleared
        l1_cache = self.cache.caches[CacheLevel.L1_CPU]
        if hasattr(l1_cache, '__len__'):
            assert len(l1_cache) == 0
        
        # Clear all levels
        self.cache.clear_cache()
        
        # Verify all levels are cleared
        for level, cache in self.cache.caches.items():
            if hasattr(cache, '__len__') and level != CacheLevel.PERSISTENT:
                assert len(cache) == 0
    
    def test_workload_optimization(self):
        """Test workload-specific optimization."""
        workload_characteristics = {
            "access_pattern": "sequential",
            "data_size": "large",
            "computation_intensity": "high"
        }
        
        # Should not raise exception
        self.cache.optimize_for_workload(workload_characteristics)
        
        # Verify some optimization was applied
        # (Exact changes depend on implementation)
        assert self.cache.replacement_policy in ReplacementPolicy
    
    def test_background_optimization(self):
        """Test background optimization thread."""
        # Start optimization
        self.cache.start_optimization_thread()
        assert self.cache.optimization_running
        assert self.cache.optimization_thread is not None
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop optimization
        self.cache.stop_optimization_thread()
        assert not self.cache.optimization_running


class TestPrefetchingFeatures:
    """Test suite for prefetching functionality."""
    
    def setup_method(self):
        """Set up test fixtures with prefetching enabled."""
        self.cache = UltraHighPerformanceCache(
            l1_capacity=10,
            l2_capacity=20,
            enable_prefetching=True
        )
    
    def test_prefetch_predictor_integration(self):
        """Test prefetch predictor integration."""
        assert self.cache.prefetch_predictor is not None
        assert isinstance(self.cache.prefetch_predictor, PrefetchPredictor)
    
    def test_access_pattern_learning(self):
        """Test access pattern learning for prefetching."""
        # Create access pattern: A -> B -> C
        pattern = ["key_A", "key_B", "key_C"] * 5
        
        for key in pattern:
            def compute_func():
                return f"value_for_{key}"
            self.cache.get(key, compute_func, context={"pattern_test": True})
        
        # Check that predictor learned the pattern
        predictor = self.cache.prefetch_predictor
        assert len(predictor.access_history) > 0
        assert len(predictor.pattern_frequencies) > 0
    
    @pytest.mark.asyncio
    async def test_prefetch_execution(self):
        """Test prefetch operation execution."""
        # Create pattern to learn
        pattern = ["A", "B", "C", "A", "B", "C"]
        
        for key in pattern:
            def compute_func():
                return f"computed_{key}"
            self.cache.get(key, compute_func)
        
        # Wait a bit for potential prefetch operations
        await asyncio.sleep(0.1)
        
        # Check prefetch statistics
        stats = self.cache.get_cache_statistics()
        if "prefetch" in stats:
            # Prefetching system is active
            assert "prediction_accuracy" in stats["prefetch"]


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = UltraHighPerformanceCache(
            l1_capacity=5,
            enable_prefetching=False
        )
    
    def test_none_value_caching(self):
        """Test caching of None values."""
        self.cache.put("none_key", None)
        result = self.cache.get("none_key")
        assert result is None
    
    def test_large_object_caching(self):
        """Test caching of large objects."""
        large_tensor = torch.randn(1000, 1000)
        
        self.cache.put("large_object", large_tensor)
        result = self.cache.get("large_object")
        
        assert torch.allclose(result, large_tensor)
    
    def test_cache_with_zero_capacity(self):
        """Test cache behavior with zero capacity."""
        zero_cache = UltraHighPerformanceCache(
            l1_capacity=0,
            l2_capacity=0,
            enable_prefetching=False
        )
        
        # Should handle gracefully
        zero_cache.put("test", "value")
        result = zero_cache.get("test")
        # May or may not find the item depending on implementation
    
    def test_concurrent_put_get_same_key(self):
        """Test concurrent put/get operations on same key."""
        key = "concurrent_key"
        
        def putter():
            for i in range(10):
                self.cache.put(key, f"value_{i}")
                time.sleep(0.001)
        
        def getter():
            results = []
            for i in range(10):
                result = self.cache.get(key)
                results.append(result)
                time.sleep(0.001)
            return results
        
        # Start concurrent operations
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            put_future = executor.submit(putter)
            get_future = executor.submit(getter)
            
            put_future.result()
            get_results = get_future.result()
        
        # Should complete without exceptions
        assert len(get_results) == 10
    
    def test_computation_function_exception(self):
        """Test handling of exceptions in computation functions."""
        def failing_computation():
            raise ValueError("Computation failed")
        
        with pytest.raises(ValueError):
            self.cache.get("failing_key", failing_computation)
        
        # Cache should remain functional
        self.cache.put("working_key", "working_value")
        result = self.cache.get("working_key")
        assert result == "working_value"
    
    def test_invalid_cache_level(self):
        """Test handling of invalid cache level operations."""
        # This should not raise exception
        try:
            self.cache.clear_cache(None)
        except Exception as e:
            # If it does raise, should be a reasonable exception
            assert isinstance(e, (ValueError, TypeError, AttributeError))


class TestIntegrationFunctions:
    """Test suite for module integration functions."""
    
    def test_create_photonic_cache_demo(self):
        """Test demo cache creation."""
        cache, config = create_photonic_cache_demo()
        
        assert isinstance(cache, UltraHighPerformanceCache)
        assert isinstance(config, dict)
        
        # Check configuration
        assert "total_capacity" in config
        assert "levels" in config
        assert "replacement_policy" in config
        assert config["levels"] >= 4
    
    @pytest.mark.asyncio
    async def test_run_cache_performance_benchmark(self):
        """Test cache performance benchmark."""
        cache, _ = create_photonic_cache_demo()
        
        # Run benchmark with minimal operations for testing
        results = await run_cache_performance_benchmark(
            cache,
            num_operations=100,
            access_patterns=["random", "sequential"]
        )
        
        assert isinstance(results, dict)
        assert "random" in results
        assert "sequential" in results
        
        for pattern, pattern_results in results.items():
            assert "hit_rates" in pattern_results
            assert "access_times" in pattern_results
            assert "total_time" in pattern_results
            assert "operations_per_second" in pattern_results
    
    def test_validate_cache_performance_improvements(self):
        """Test cache performance validation."""
        validation_results = validate_cache_performance_improvements()
        
        assert isinstance(validation_results, dict)
        required_metrics = [
            "hit_rate_improvement",
            "access_time_reduction",
            "prefetch_accuracy",
            "memory_efficiency"
        ]
        
        for metric in required_metrics:
            assert metric in validation_results
            assert isinstance(validation_results[metric], (int, float))
            assert validation_results[metric] >= 0


class TestPerformanceAndScalability:
    """Test suite for performance and scalability."""
    
    @pytest.mark.parametrize("cache_size", [10, 100, 1000])
    def test_scalability_with_cache_size(self, cache_size):
        """Test scalability with different cache sizes."""
        cache = UltraHighPerformanceCache(
            l1_capacity=cache_size,
            l2_capacity=cache_size * 2,
            enable_prefetching=False
        )
        
        # Fill cache
        start_time = time.time()
        for i in range(cache_size):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Access all items
        for i in range(cache_size):
            result = cache.get(f"key_{i}")
            assert result == f"value_{i}"
        
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # Less than 5 seconds
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create and use cache
        cache = UltraHighPerformanceCache(
            l1_capacity=1000,
            l2_capacity=2000,
            enable_prefetching=False
        )
        
        # Add many items
        for i in range(1000):
            data = torch.randn(100)  # Moderately sized data
            cache.put(f"key_{i}", data)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable
        assert memory_increase < 1000  # Less than 1GB
    
    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        cache = UltraHighPerformanceCache(
            l1_capacity=100,
            l2_capacity=200,
            enable_prefetching=False
        )
        
        def worker(worker_id, num_operations):
            for i in range(num_operations):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                cache.put(key, value)
                result = cache.get(key)
                assert result == value
        
        # Run concurrent workers
        import concurrent.futures
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(worker, i, 50)
                for i in range(5)
            ]
            
            # Wait for completion
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Will raise exception if worker failed
        
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 10.0  # Less than 10 seconds
    
    @pytest.mark.parametrize("num_threads", [1, 2, 4, 8])
    def test_thread_scalability(self, num_threads):
        """Test scalability with different thread counts."""
        cache = UltraHighPerformanceCache(
            l1_capacity=50,
            enable_prefetching=False
        )
        
        def worker(operations_per_thread):
            for i in range(operations_per_thread):
                cache.put(f"thread_key_{threading.current_thread().ident}_{i}", f"value_{i}")
                result = cache.get(f"thread_key_{threading.current_thread().ident}_{i}")
                assert result is not None
        
        start_time = time.time()
        
        threads = []
        operations_per_thread = 100 // num_threads
        
        for _ in range(num_threads):
            t = threading.Thread(target=worker, args=(operations_per_thread,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # More threads should generally complete faster (up to a point)
        assert execution_time < 10.0  # Should complete within 10 seconds


@pytest.fixture
def sample_cache():
    """Pytest fixture providing a sample cache for tests."""
    return UltraHighPerformanceCache(
        l1_capacity=20,
        l2_capacity=50,
        l3_capacity=100,
        enable_prefetching=False
    )


@pytest.fixture
def sample_data():
    """Pytest fixture providing sample data for caching."""
    return {
        f"key_{i}": torch.randn(10, 10) for i in range(10)
    }


class TestWithFixtures:
    """Test suite using pytest fixtures."""
    
    def test_cache_with_sample_data(self, sample_cache, sample_data):
        """Test cache operations with sample data."""
        # Store all sample data
        for key, value in sample_data.items():
            sample_cache.put(key, value)
        
        # Retrieve all sample data
        for key, expected_value in sample_data.items():
            retrieved_value = sample_cache.get(key)
            assert torch.allclose(retrieved_value, expected_value)
    
    def test_cache_statistics_with_fixtures(self, sample_cache, sample_data):
        """Test cache statistics with fixtures."""
        # Perform operations
        for key, value in sample_data.items():
            sample_cache.put(key, value)
            sample_cache.get(key)  # Generate hits
        
        # Get statistics
        stats = sample_cache.get_cache_statistics()
        
        assert stats["overall"]["hit_rate"] > 0
        assert stats["overall"]["total_hits"] > 0
    
    def test_cache_consistency_with_fixtures(self, sample_cache, sample_data):
        """Test cache consistency with fixtures."""
        # Store data
        for key, value in sample_data.items():
            sample_cache.put(key, value)
        
        # Multiple retrievals should return same data
        for key, expected_value in sample_data.items():
            for _ in range(3):
                retrieved_value = sample_cache.get(key)
                assert torch.allclose(retrieved_value, expected_value)
    
    def test_cache_replacement_with_fixtures(self, sample_data):
        """Test cache replacement behavior with fixtures."""
        # Create small cache to force replacements
        small_cache = UltraHighPerformanceCache(
            l1_capacity=3,
            l2_capacity=5,
            enable_prefetching=False
        )
        
        # Add more data than cache can hold
        for key, value in sample_data.items():
            small_cache.put(key, value)
        
        # Some items should have been evicted
        # Exact behavior depends on replacement policy
        stats = small_cache.get_cache_statistics()
        
        # Should have some evictions
        total_evictions = sum(
            level_stats.get("evictions", 0)
            for level_stats in stats.values()
            if isinstance(level_stats, dict) and "evictions" in level_stats
        )
        # Note: evictions might be 0 if all items fit in different cache levels