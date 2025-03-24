from abc import ABC, abstractmethod

import numpy as np


class FusionStrategy(ABC):
    """Abstract base class for emotion fusion strategies."""

    @abstractmethod
    def fuse(
        self, predictions: dict[str, dict[str, float]]
    ) -> tuple[str, dict[str, float]]:
        """Fuse emotions from different modalities."""
        pass


class AverageFusion(FusionStrategy):
    """Simple averaging fusion strategy."""

    def fuse(
        self, predictions: dict[str, dict[str, float]]
    ) -> tuple[str, dict[str, float]]:
        if not predictions:
            raise ValueError("No predictions to fuse")

        emotion_labels = list(next(iter(predictions.values())).keys())
        scores = np.array([list(pred.values()) for pred in predictions.values()])
        avg_scores = np.mean(scores, axis=0)

        top_emotion = emotion_labels[np.argmax(avg_scores)]
        final_scores = dict(zip(emotion_labels, avg_scores, strict=False))

        return top_emotion, final_scores


class WeightedFusion(FusionStrategy):
    """Weighted fusion strategy."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def fuse(
        self, predictions: dict[str, dict[str, float]]
    ) -> tuple[str, dict[str, float]]:
        if not predictions:
            raise ValueError("No predictions to fuse")

        emotion_labels = list(next(iter(predictions.values())).keys())
        weighted_scores = np.zeros(len(emotion_labels))
        total_weight = 0

        for modality, scores in predictions.items():
            if modality in self.weights:
                weight = self.weights[modality]
                weighted_scores += weight * np.array(list(scores.values()))
                total_weight += weight

        if total_weight > 0:
            weighted_scores /= total_weight

        top_emotion = emotion_labels[np.argmax(weighted_scores)]
        final_scores = dict(zip(emotion_labels, weighted_scores, strict=False))

        return top_emotion, final_scores
