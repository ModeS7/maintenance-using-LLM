"""
Inference module for turbofan engine RUL prediction.

Loads trained model and provides prediction interface for the demo.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

from src.model import create_model
from src.data_loader import (
    CMAPSSDataLoader,
    EngineData,
    FEATURE_COLUMNS,
    SENSOR_NAMES,
    SENSOR_UNITS,
    SENSORS_TO_DROP,
    SETTING_NAMES,
    MAX_RUL,
    check_data_available,
    print_download_instructions,
)


def get_severity(rul: int) -> str:
    """
    Categorize RUL into severity levels.

    - Critical: RUL < 30 cycles
    - Warning: 30 <= RUL < 60
    - Caution: 60 <= RUL < 90
    - Healthy: RUL >= 90
    """
    if rul < 30:
        return "critical"
    elif rul < 60:
        return "warning"
    elif rul < 90:
        return "caution"
    else:
        return "healthy"


def get_severity_description(severity: str) -> str:
    """Get human-readable severity description."""
    descriptions = {
        "critical": "Immediate maintenance required - high failure risk",
        "warning": "Schedule maintenance soon - degradation detected",
        "caution": "Monitor closely - early signs of wear",
        "healthy": "Normal operation - no immediate concerns",
    }
    return descriptions.get(severity, "Unknown status")


@dataclass
class RULPrediction:
    """Container for RUL prediction results."""
    engine_id: int
    cycle: int
    predicted_rul: int
    true_rul: Optional[int]
    severity: str
    severity_description: str


class RULInference:
    """
    Inference engine for turbofan RUL predictions.

    Handles model loading, prediction, and result formatting.
    """

    def __init__(
        self,
        model_path: str = "models/rul_model.pt",
        data_dir: str = "data/CMAPSSData",
        window_size: int = 30,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.data_dir = data_dir
        self.window_size = window_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[torch.nn.Module] = None
        self.data_loader: Optional[CMAPSSDataLoader] = None
        self._is_loaded = False

    def load(self) -> bool:
        """
        Load the model and data.

        Returns:
            True if loading successful
        """
        # Check data
        if not check_data_available(self.data_dir):
            print("Dataset not found!")
            print_download_instructions()
            return False

        # Load data
        self.data_loader = CMAPSSDataLoader(data_dir=self.data_dir)
        if not self.data_loader.load():
            return False

        self.data_loader.create_train_demo_split()
        self.data_loader.compute_normalization_stats()

        # Load model
        if not self.model_path.exists():
            print(f"Warning: Model not found at {self.model_path}")
            print("Predictions will use ground truth from dataset.")
            self._is_loaded = True
            return True

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            n_features = len(FEATURE_COLUMNS)
            self.model = create_model(
                model_type="lstm",
                input_size=n_features,
                hidden_size=64,
                num_layers=2,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            val_metrics = checkpoint.get("val_metrics", {})
            print(f"Model loaded from {self.model_path}")
            print(f"Validation RMSE: {val_metrics.get('rmse', 'N/A'):.2f}")

            self._is_loaded = True
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self._is_loaded = True
            return True

    def predict(self, engine_id: int, at_cycle: Optional[int] = None) -> Optional[RULPrediction]:
        """
        Predict RUL for an engine at a specific cycle.

        Args:
            engine_id: Engine ID
            at_cycle: Cycle number (None = last available cycle)

        Returns:
            RULPrediction or None if engine not found
        """
        if not self._is_loaded:
            self.load()

        engine = self.data_loader.get_engine(engine_id)
        if engine is None:
            return None

        # Determine actual cycle
        if at_cycle is None:
            at_cycle = int(engine.max_cycle)
        at_cycle = min(at_cycle, int(engine.max_cycle))

        # Get sequence and true RUL
        result = self.data_loader.get_engine_sequence(engine_id, at_cycle, self.window_size)
        if result is None:
            return None

        sequence, true_rul = result

        if self.model is not None:
            # Use model prediction
            sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            predicted_rul = int(self.model.predict(sequence_tensor))
        else:
            # Use ground truth
            predicted_rul = true_rul

        # Clamp to valid range (no upper limit if MAX_RUL is None)
        predicted_rul = max(0, predicted_rul)
        if MAX_RUL is not None:
            predicted_rul = min(predicted_rul, MAX_RUL)

        severity = get_severity(predicted_rul)

        return RULPrediction(
            engine_id=engine_id,
            cycle=at_cycle,
            predicted_rul=predicted_rul,
            true_rul=true_rul,
            severity=severity,
            severity_description=get_severity_description(severity),
        )

    def get_engine_status(self, engine_id: int, at_cycle: Optional[int] = None) -> Optional[Dict]:
        """
        Get comprehensive status for an engine.

        This is the main interface used by LLM tools.
        """
        if not self._is_loaded:
            self.load()

        engine = self.data_loader.get_engine(engine_id)
        if engine is None:
            return None

        prediction = self.predict(engine_id, at_cycle)
        if prediction is None:
            return None

        # Calculate life remaining as percentage of engine's total lifecycle
        life_remaining_pct = round(prediction.predicted_rul / engine.max_cycle * 100, 1) if engine.max_cycle > 0 else 0

        return {
            "engine_id": engine_id,
            "dataset": engine.dataset,
            "current_cycle": prediction.cycle,
            "max_cycle": engine.max_cycle,
            "total_cycles": engine.n_cycles,
            "predicted_rul": prediction.predicted_rul,
            "true_rul": prediction.true_rul,
            "severity": prediction.severity,
            "severity_description": prediction.severity_description,
            "life_remaining_pct": life_remaining_pct,
        }

    def get_sensor_readings(self, engine_id: int, at_cycle: Optional[int] = None) -> Optional[Dict]:
        """Get sensor readings for an engine at a specific cycle."""
        if not self._is_loaded:
            self.load()

        engine = self.data_loader.get_engine(engine_id)
        if engine is None:
            return None

        # Get state at cycle
        state = self.data_loader.get_engine_at_cycle(engine_id, at_cycle or engine.max_cycle)
        if state is None:
            return None

        # Get sensor stats for context
        stats = self.data_loader.get_sensor_stats()

        readings = {}
        for i in range(21):
            sensor_name = f"sensor_{i+1}"
            value = float(state["sensors"][i])
            stat = stats[sensor_name]

            # Determine if reading is abnormal (beyond 2 std from mean)
            z_score = (value - stat["mean"]) / (stat["std"] + 1e-8)
            is_abnormal = abs(z_score) > 2

            # Skip dropped sensors in feature set
            is_informative = sensor_name not in SENSORS_TO_DROP

            readings[SENSOR_NAMES[sensor_name]] = {
                "sensor_id": sensor_name,
                "value": round(value, 2),
                "unit": SENSOR_UNITS[sensor_name],
                "mean": round(stat["mean"], 2),
                "std": round(stat["std"], 2),
                "z_score": round(z_score, 2),
                "is_abnormal": is_abnormal,
                "is_informative": is_informative,
                "status": "abnormal" if is_abnormal else "normal",
            }

        # Add operational settings
        settings = {}
        for i in range(3):
            setting_name = f"setting_{i+1}"
            settings[SETTING_NAMES[setting_name]] = round(float(state["settings"][i]), 4)

        return {
            "engine_id": engine_id,
            "cycle": state["cycle"],
            "rul": state["rul"],
            "readings": readings,
            "settings": settings,
        }

    def list_demo_engines(
        self,
        severity_filter: Optional[str] = None,
        at_cycle_pct: float = 0.7,
    ) -> List[Dict]:
        """
        List all demo engines with their status.

        Args:
            severity_filter: Filter by severity ("critical", "warning", "caution", "healthy")
            at_cycle_pct: Evaluate engines at this percentage of their lifecycle (0.7 = 70%)

        Returns:
            List of engine status dictionaries
        """
        if not self._is_loaded:
            self.load()

        results = []

        for engine_id in self.data_loader.demo_ids:
            engine = self.data_loader.get_engine(engine_id)
            if engine is None:
                continue

            # Evaluate at a percentage of lifecycle (not at end/failure)
            eval_cycle = int(engine.max_cycle * at_cycle_pct)
            eval_cycle = max(self.window_size, eval_cycle)  # Need at least window_size

            prediction = self.predict(engine_id, at_cycle=eval_cycle)
            if prediction is None:
                continue

            # Apply filter
            if severity_filter and prediction.severity != severity_filter:
                continue

            results.append({
                "engine_id": engine_id,
                "dataset": engine.dataset,
                "current_cycle": prediction.cycle,
                "total_cycles": engine.n_cycles,
                "predicted_rul": prediction.predicted_rul,
                "true_rul": prediction.true_rul,
                "severity": prediction.severity,
            })

        # Sort by predicted RUL (most critical first)
        results.sort(key=lambda x: x["predicted_rul"])

        return results

    def get_fleet_summary(self) -> Dict:
        """Get aggregate statistics for the demo fleet."""
        if not self._is_loaded:
            self.load()

        engines = self.list_demo_engines()

        total = len(engines)
        critical = sum(1 for e in engines if e["severity"] == "critical")
        warning = sum(1 for e in engines if e["severity"] == "warning")
        caution = sum(1 for e in engines if e["severity"] == "caution")
        healthy = sum(1 for e in engines if e["severity"] == "healthy")

        # Average RUL
        avg_rul = np.mean([e["predicted_rul"] for e in engines]) if engines else 0

        # Engines needing immediate attention
        attention_needed = [
            {
                "engine_id": e["engine_id"],
                "predicted_rul": e["predicted_rul"],
                "severity": e["severity"],
            }
            for e in engines
            if e["severity"] in ["critical", "warning"]
        ][:10]

        # Dataset breakdown
        dataset_counts = {}
        for e in engines:
            ds = e["dataset"]
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

        return {
            "total_engines": total,
            "critical": critical,
            "warning": warning,
            "caution": caution,
            "healthy": healthy,
            "average_rul": round(avg_rul, 1),
            "fleet_health_pct": round(healthy / total * 100, 1) if total > 0 else 0,
            "immediate_attention": attention_needed,
            "dataset_breakdown": dataset_counts,
        }

    def get_engine_timeline(self, engine_id: int, step: int = 10) -> Optional[List[Dict]]:
        """
        Get RUL predictions over the engine's lifecycle.

        Args:
            engine_id: Engine ID
            step: Cycle step size

        Returns:
            List of predictions at different cycles
        """
        if not self._is_loaded:
            self.load()

        engine = self.data_loader.get_engine(engine_id)
        if engine is None:
            return None

        timeline = []
        for cycle in range(self.window_size, engine.max_cycle + 1, step):
            prediction = self.predict(engine_id, cycle)
            if prediction:
                timeline.append({
                    "cycle": cycle,
                    "predicted_rul": prediction.predicted_rul,
                    "true_rul": prediction.true_rul,
                    "severity": prediction.severity,
                })

        return timeline


# Global inference instance
_inference_instance: Optional[RULInference] = None


def get_inference() -> RULInference:
    """Get or create the global inference instance."""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = RULInference()
        _inference_instance.load()
    return _inference_instance


if __name__ == "__main__":
    # Test inference
    print("Testing RUL Inference...\n")

    inference = RULInference()
    inference.load()

    # Test fleet summary
    print("Fleet Summary:")
    summary = inference.get_fleet_summary()
    print(f"  Total engines: {summary['total_engines']}")
    print(f"  Critical: {summary['critical']}")
    print(f"  Warning: {summary['warning']}")
    print(f"  Caution: {summary['caution']}")
    print(f"  Healthy: {summary['healthy']}")
    print(f"  Average RUL: {summary['average_rul']:.1f} cycles")

    # Test single engine
    demo_engines = inference.data_loader.demo_ids
    if demo_engines:
        engine_id = demo_engines[0]
        print(f"\nEngine {engine_id} Status:")
        status = inference.get_engine_status(engine_id)
        if status:
            print(f"  Dataset: {status['dataset']}")
            print(f"  Current cycle: {status['current_cycle']}")
            print(f"  Predicted RUL: {status['predicted_rul']} cycles")
            print(f"  True RUL: {status['true_rul']} cycles")
            print(f"  Severity: {status['severity']}")
            print(f"  Life remaining: {status['life_remaining_pct']:.1f}%")

        # Test sensor readings
        print(f"\n  Sensor readings (abnormal only):")
        readings = inference.get_sensor_readings(engine_id)
        if readings:
            for name, r in readings["readings"].items():
                if r["is_abnormal"]:
                    print(f"    {name}: {r['value']} {r['unit']} (z={r['z_score']:.1f})")
