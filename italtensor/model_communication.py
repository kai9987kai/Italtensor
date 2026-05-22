"""Multi-model panel deliberation, consensus, and communication logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Sequence

import numpy as np

from .modeling import predict_probability
from .preprocessing import FeatureStandardizer

FUSION_MEAN = "mean"
FUSION_MEDIAN = "median"
FUSION_VOTE = "vote"
FUSION_WEIGHTED = "weighted"
FUSION_STACKING = "stacking"
FUSION_CHOICES = (FUSION_MEAN, FUSION_MEDIAN, FUSION_VOTE, FUSION_WEIGHTED, FUSION_STACKING)


@dataclass(frozen=True)
class ModelMessage:
    """One line in the model communication log."""

    sender: str
    kind: str
    payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "kind": self.kind,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMessage:
        return cls(
            sender=str(data.get("sender", "unknown")),
            kind=str(data.get("kind", "info")),
            payload=data.get("payload") if isinstance(data.get("payload"), dict) else {},
            timestamp=str(data.get("timestamp", datetime.now(UTC).isoformat())),
        )


@dataclass
class PanelMember:
    name: str
    model: Any
    preprocessor: FeatureStandardizer | None = None
    weight: float = 1.0
    threshold: float = 0.5


@dataclass
class PanelPrediction:
    per_model: dict[str, np.ndarray]
    consensus: np.ndarray
    disagreement: np.ndarray
    fusion: str
    messages: list[ModelMessage]


class ModelPanel:
    """Coordinate multiple models on the same input."""

    def __init__(
        self,
        members: list[PanelMember],
        *,
        fusion: str = FUSION_MEAN,
        stacking_coef: np.ndarray | None = None,
    ):
        if not members:
            raise ValueError("Model panel requires at least one member.")
        fusion_norm = (fusion or FUSION_MEAN).lower().strip()
        if fusion_norm not in FUSION_CHOICES:
            raise ValueError(f"fusion must be one of: {', '.join(FUSION_CHOICES)}.")
        self.members = members
        self.fusion = fusion_norm
        self.stacking_coef = stacking_coef

    @property
    def input_dim(self) -> int:
        member = self.members[0]
        if member.preprocessor is not None:
            return int(member.preprocessor.mean.shape[0])
        return int(getattr(member.model, "input_dim", 0))

    def predict(self, samples: Sequence[float] | np.ndarray, *, threshold: float = 0.5) -> PanelPrediction:
        sample_array = np.asarray(samples, dtype=np.float32)
        if sample_array.ndim == 1:
            sample_array = sample_array.reshape(1, -1)

        per_model: dict[str, np.ndarray] = {}
        matrix: list[np.ndarray] = []
        messages: list[ModelMessage] = []

        for member in self.members:
            prepared = (
                member.preprocessor.transform(sample_array)
                if member.preprocessor is not None
                else sample_array
            )
            probs = predict_probability(member.model, prepared).reshape(-1)
            per_model[member.name] = probs
            matrix.append(probs)
            label = 1 if float(probs[0]) >= member.threshold else 0
            messages.append(
                ModelMessage(
                    sender=member.name,
                    kind="vote",
                    payload={
                        "probability": float(probs[0]),
                        "label": label,
                        "threshold": member.threshold,
                    },
                )
            )

        stacked = np.stack(matrix, axis=0)
        disagreement = np.std(stacked, axis=0).astype(np.float32)
        consensus = self._fuse(stacked, threshold=threshold)

        messages.append(
            ModelMessage(
                sender="panel",
                kind="consensus",
                payload={
                    "fusion": self.fusion,
                    "probability": float(consensus[0]),
                    "disagreement": float(disagreement[0]),
                    "label": 1 if float(consensus[0]) >= threshold else 0,
                },
            )
        )
        return PanelPrediction(
            per_model=per_model,
            consensus=consensus,
            disagreement=disagreement,
            fusion=self.fusion,
            messages=messages,
        )

    def _fuse(self, stacked: np.ndarray, *, threshold: float) -> np.ndarray:
        if self.fusion == FUSION_MEDIAN:
            return np.median(stacked, axis=0).astype(np.float32)
        if self.fusion == FUSION_VOTE:
            votes = (stacked >= threshold).astype(np.float32)
            return np.mean(votes, axis=0).astype(np.float32)
        if self.fusion == FUSION_WEIGHTED:
            weights = np.array([max(1e-6, m.weight) for m in self.members], dtype=np.float32)
            weights /= weights.sum()
            return np.average(stacked, axis=0, weights=weights).astype(np.float32)
        if self.fusion == FUSION_STACKING and self.stacking_coef is not None:
            meta = stacked.T
            logits = meta @ self.stacking_coef.reshape(-1, 1)
            return _sigmoid(logits.reshape(-1)).astype(np.float32)
        return np.mean(stacked, axis=0).astype(np.float32)

    def format_deliberation(self, prediction: PanelPrediction) -> str:
        lines = [f"Panel fusion={prediction.fusion}, disagreement={float(prediction.disagreement[0]):.4f}"]
        for message in prediction.messages:
            if message.kind == "vote":
                lines.append(
                    f"  {message.sender}: p={message.payload['probability']:.4f} "
                    f"label={message.payload['label']} (thr={message.payload['threshold']:.2f})"
                )
            elif message.kind == "consensus":
                lines.append(
                    f"  consensus: p={message.payload['probability']:.4f} "
                    f"label={message.payload['label']}"
                )
        return "\n".join(lines)


def fit_stacking_weights(
    members: list[PanelMember],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Fit linear stacking weights on validation features (Wolpert-style stacking)."""
    if not members:
        raise ValueError("Cannot fit stacking weights without panel members.")
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    meta_cols: list[np.ndarray] = []
    for member in members:
        prepared = member.preprocessor.transform(features) if member.preprocessor is not None else features
        meta_cols.append(predict_probability(member.model, prepared).reshape(-1))
    meta = np.stack(meta_cols, axis=1).astype(np.float32)
    ones = np.ones((meta.shape[0], 1), dtype=np.float32)
    design = np.hstack([meta, ones])
    reg = ridge * np.eye(design.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return coef.astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))
