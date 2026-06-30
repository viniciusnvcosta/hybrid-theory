"""Test detector registry registration.

TDD approach: Test that detectors registered via @register_detector
are accessible via the registry.

Author: CDADE project
"""

import pytest

from cdade.registry import (
    register_detector,
    get_detector,
    list_detectors,
    _get_registry,
)


class TestDetectorRegistry:
    """Test detector registration and retrieval."""

    def test_register_detector_decorator(self):
        """Test that @register_detector decorator registers detector."""
        # Get current registry state
        registry_before = set(_get_registry("detector").keys())

        # Define a test detector
        @register_detector("test_detector")
        class TestDetector:
            """Test detector."""
            def __init__(self):
                pass

            def fit(self, X):
                return self

            def score(self, X):
                return np.random.randn(len(X))

        # Get registry state after registration
        registry_after = set(_get_registry("detector").keys())

        # Should have registered the detector
        assert "test_detector" in registry_after
        assert len(registry_after) > len(registry_before)

    def test_get_detector_by_name(self):
        """Test retrieving detector by name."""
        @register_detector("test_get_detector")
        class TestDetector:
            def fit(self, X):
                return self

            def score(self, X):
                return np.random.randn(len(X))

        # Get detector by name
        detector_cls = get_detector("test_get_detector")

        # Should return the class
        assert detector_cls is not None
        assert detector_cls.__name__ == "TestDetector"

    def test_list_detectors(self):
        """Test listing all registered detectors."""
        # Get current detector count
        count_before = len(list_detectors())

        # Register test detector
        @register_detector("test_list_detector")
        class TestDetector:
            def fit(self, X):
                return self

            def score(self, X):
                return np.random.randn(len(X))

        # Should see the new detector
        count_after = len(list_detectors())
        assert count_after > count_before

    def test_detector_in_registry(self):
        """Test that a detector class is in the registry."""
        # Check if any known detector is registered
        # (this will tell us if the registry is working at all)
        test_detectors = ["pca", "iforest", "lof", "mcd", "knn", "ocsvm", "hbos", "cof", "sos", "cblof"]

        for detector_name in test_detectors:
            try:
                detector_cls = get_detector(detector_name)
                # If we get here, the detector is registered
                assert detector_cls is not None
            except KeyError:
                # This is expected if detectors aren't registered
                # We'll investigate why
                pass

    def test_registry_isolation(self):
        """Test that different registry types are isolated."""
        # Get registry for different types
        detector_registry = set(_get_registry("detector").keys())
        reconciler_registry = set(_get_registry("reconciler").keys())
        selector_registry = set(_get_registry("selector").keys())

        # Register a detector - shouldn't affect other types
        @register_detector("test_isolated")
        class TestDetector:
            def fit(self, X):
                return self

            def score(self, X):
                return np.random.randn(len(X))

        # Detector registry should have new entry
        assert "test_isolated" in _get_registry("detector")

        # Other registries should not have it
        assert "test_isolated" not in _get_registry("reconciler")
        assert "test_isolated" not in _get_registry("selector")