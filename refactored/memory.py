import numpy as np
import sspspace
from collections import deque


class WorkingMemoryManager:
    """Enhanced working memory management with quality-based filtering"""

    def __init__(self, memory_size=15, similarity_threshold=0.85, gamma=0.9, min_attention_threshold=100):
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.gamma = gamma
        self.min_attention_threshold = min_attention_threshold

        # Memory buffers
        self.feature_buffer = deque(maxlen=memory_size)
        self.coord_buffer = deque(maxlen=memory_size)
        self.quality_scores = deque(maxlen=memory_size)
        self.frame_indices = deque(maxlen=memory_size)

        # Statistics
        self.total_frames = 0
        self.stored_frames = 0

    def calculate_feature_quality(self, roi_patch, attention_strength, patch_variance=None):
        """Calculate quality score for a feature based on multiple factors"""
        # Normalize attention strength (0-255 -> 0-1)
        attention_score = min(attention_strength / 255.0, 1.0)

        # Calculate patch informativeness
        if patch_variance is None:
            patch_variance = np.var(roi_patch)

        # Normalize variance score (higher variance = more informative)
        variance_score = min(patch_variance / 1000.0, 1.0)

        # Check if patch has sufficient detail (not too uniform)
        detail_score = 1.0 if patch_variance > 100 else 0.5

        # Combined quality score
        quality = (attention_score * 0.5 + variance_score * 0.3 + detail_score * 0.2)

        return quality

    def is_significantly_different(self, new_features, threshold=None):
        """Check if new features are significantly different from recent memory"""
        if not self.feature_buffer:
            return True

        threshold = threshold or self.similarity_threshold

        # Compare with recent features (last 3-5 frames)
        recent_count = min(5, len(self.feature_buffer))
        recent_features = list(self.feature_buffer)[-recent_count:]

        for recent_features_vec in recent_features:
            # Cosine similarity
            similarity = np.dot(new_features, recent_features_vec) / (
                    np.linalg.norm(new_features) * np.linalg.norm(recent_features_vec) + 1e-8
            )
            if similarity > threshold:
                return False
        return True

    def should_store_memory(self, roi_patch, attention_strength, new_features, frame_idx):
        """Determine if this frame should be stored in working memory"""
        self.total_frames += 1

        # Calculate quality metrics
        patch_variance = np.var(roi_patch)
        quality_score = self.calculate_feature_quality(roi_patch, attention_strength, patch_variance)

        # Multiple criteria for storage
        criteria = {
            'quality_threshold': quality_score > 0.4,
            'attention_threshold': attention_strength > self.min_attention_threshold,
            'novelty_check': self.is_significantly_different(new_features),
            'patch_variance': patch_variance > 50,  # Ensure patch isn't too uniform
        }

        # Store if most criteria are met
        should_store = sum(criteria.values()) >= 3

        if should_store:
            self.stored_frames += 1

        return should_store, quality_score, criteria

    def update_memory(self, features, coords, quality_score, frame_idx):
        """Update working memory with new high-quality features"""
        self.feature_buffer.append(features.copy())
        self.coord_buffer.append(coords.copy())
        self.quality_scores.append(quality_score)
        self.frame_indices.append(frame_idx)

    def get_consolidated_memory(self, coord_encoder):
        """Get consolidated memory representation using quality weighting"""
        if not self.feature_buffer:
            return coord_encoder.encode([[0, 0]])

        # Convert to arrays
        features_array = np.array(self.feature_buffer)
        coords_array = np.array(self.coord_buffer)
        weights = np.array(self.quality_scores)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Weighted average of features and coordinates
        consolidated_features = np.average(features_array, axis=0, weights=weights)
        consolidated_coords = np.average(coords_array, axis=0, weights=weights)

        # Ensure features are normalized
        consolidated_features = consolidated_features / (np.linalg.norm(consolidated_features) + 1e-8)

        # Combine spatial and feature information using SSP
        roi_center = coord_encoder.encode([[consolidated_coords[0], consolidated_coords[1]]])
        img_feat_ssp = sspspace.SSP(consolidated_features)

        return roi_center * img_feat_ssp

    def get_statistics(self):
        """Get memory storage statistics"""
        storage_rate = (self.stored_frames / self.total_frames) if self.total_frames > 0 else 0
        return {
            'total_frames': self.total_frames,
            'stored_frames': self.stored_frames,
            'storage_rate': storage_rate,
            'buffer_size': len(self.feature_buffer),
            'avg_quality': np.mean(self.quality_scores) if self.quality_scores else 0
        }
