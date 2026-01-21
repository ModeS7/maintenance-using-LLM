"""
Data loader for NASA C-MAPSS turbofan engine degradation dataset.

Loads the text-based run-to-failure simulation data and prepares it for
LSTM-based RUL (Remaining Useful Life) prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# Column names for the dataset
COLUMN_NAMES = (
    ["unit", "cycle"] +
    [f"setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# Sensor names and descriptions
SENSOR_NAMES = {
    "sensor_1": "T2 (Fan inlet temp)",
    "sensor_2": "T24 (LPC outlet temp)",
    "sensor_3": "T30 (HPC outlet temp)",
    "sensor_4": "T50 (LPT outlet temp)",
    "sensor_5": "P2 (Fan inlet pressure)",
    "sensor_6": "P15 (Bypass duct pressure)",
    "sensor_7": "P30 (HPC outlet pressure)",
    "sensor_8": "Nf (Fan speed)",
    "sensor_9": "Nc (Core speed)",
    "sensor_10": "epr (Engine pressure ratio)",
    "sensor_11": "Ps30 (HPC static pressure)",
    "sensor_12": "phi (Fuel flow ratio)",
    "sensor_13": "NRf (Corrected fan speed)",
    "sensor_14": "NRc (Corrected core speed)",
    "sensor_15": "BPR (Bypass ratio)",
    "sensor_16": "farB (Fuel-air ratio)",
    "sensor_17": "htBleed (Bleed enthalpy)",
    "sensor_18": "Nf_dmd (Demanded fan speed)",
    "sensor_19": "PCNfR_dmd (Demanded corrected fan speed)",
    "sensor_20": "W31 (HPT coolant bleed)",
    "sensor_21": "W32 (LPT coolant bleed)",
}

# Sensor units
SENSOR_UNITS = {
    "sensor_1": "°R", "sensor_2": "°R", "sensor_3": "°R", "sensor_4": "°R",
    "sensor_5": "psia", "sensor_6": "psia", "sensor_7": "psia",
    "sensor_8": "rpm", "sensor_9": "rpm",
    "sensor_10": "-", "sensor_11": "psia", "sensor_12": "pps/psi",
    "sensor_13": "rpm", "sensor_14": "rpm", "sensor_15": "-", "sensor_16": "-",
    "sensor_17": "-", "sensor_18": "rpm", "sensor_19": "rpm",
    "sensor_20": "lbm/s", "sensor_21": "lbm/s",
}

# Operational settings
SETTING_NAMES = {
    "setting_1": "Altitude",
    "setting_2": "Mach number",
    "setting_3": "Throttle resolver angle",
}

# Sensors to drop (constant or nearly constant in FD001/FD003)
# These sensors provide no useful information for prediction
SENSORS_TO_DROP = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

# Feature columns for model input (14 sensors)
FEATURE_COLUMNS = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" not in SENSORS_TO_DROP]

# Dataset subsets
DATASET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": 1, "train_engines": 100, "test_engines": 100},
    "FD002": {"conditions": 6, "fault_modes": 1, "train_engines": 260, "test_engines": 259},
    "FD003": {"conditions": 1, "fault_modes": 2, "train_engines": 100, "test_engines": 100},
    "FD004": {"conditions": 6, "fault_modes": 2, "train_engines": 248, "test_engines": 249},
}

# Maximum RUL cap (None = no cap, use actual RUL)
MAX_RUL = None


@dataclass
class EngineData:
    """Data for a single engine."""
    unit_id: int
    dataset: str
    cycles: np.ndarray  # Cycle numbers
    settings: np.ndarray  # Operational settings (n_cycles, 3)
    sensors: np.ndarray  # Sensor readings (n_cycles, 21)
    rul: np.ndarray  # RUL at each cycle
    max_cycle: int

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)


class CMAPSSDataLoader:
    """
    Data loader for C-MAPSS dataset.

    Loads all four subsets and combines them with unique engine IDs.
    """

    def __init__(
        self,
        data_dir: str = "data/CMAPSSData",
        datasets: Optional[List[str]] = None,
        max_rul: Optional[int] = MAX_RUL,
    ):
        self.data_dir = data_dir
        self.datasets = datasets or ["FD001", "FD002", "FD003", "FD004"]
        self.max_rul = max_rul  # None means no cap

        # Internal state
        self._engines: Dict[int, EngineData] = {}
        self._train_ids: List[int] = []
        self._demo_ids: List[int] = []
        self._normalization_stats: Optional[Dict] = None
        self._is_loaded: bool = False

    @property
    def train_ids(self) -> List[int]:
        return self._train_ids

    @property
    def demo_ids(self) -> List[int]:
        return self._demo_ids

    def load(self) -> bool:
        """Load all datasets."""
        if self._is_loaded:
            return True

        data_path = Path(self.data_dir)
        if not data_path.exists():
            print(f"Error: Data directory not found: {data_path}")
            return False

        engine_id_offset = 0

        for dataset in self.datasets:
            train_file = data_path / f"train_{dataset}.txt"
            if not train_file.exists():
                print(f"Warning: {train_file} not found, skipping {dataset}")
                continue

            # Load training data
            df = pd.read_csv(
                train_file,
                sep=r"\s+",
                header=None,
                names=COLUMN_NAMES,
            )

            # Process each engine
            for unit in df["unit"].unique():
                engine_df = df[df["unit"] == unit].copy()
                engine_df = engine_df.sort_values("cycle")

                cycles = engine_df["cycle"].values
                max_cycle = cycles.max()

                # Calculate RUL (optionally capped at max_rul)
                rul = max_cycle - cycles
                if self.max_rul is not None:
                    rul = np.clip(rul, 0, self.max_rul)

                # Extract features
                settings = engine_df[[f"setting_{i}" for i in range(1, 4)]].values
                sensors = engine_df[[f"sensor_{i}" for i in range(1, 22)]].values

                # Create unique engine ID
                engine_id = engine_id_offset + int(unit)

                self._engines[engine_id] = EngineData(
                    unit_id=engine_id,
                    dataset=dataset,
                    cycles=cycles,
                    settings=settings,
                    sensors=sensors,
                    rul=rul,
                    max_cycle=max_cycle,
                )

            n_engines = len(df["unit"].unique())
            engine_id_offset += n_engines
            print(f"Loaded {dataset}: {n_engines} engines")

        print(f"Total engines loaded: {len(self._engines)}")
        self._is_loaded = True
        return True

    def create_train_demo_split(self, demo_ratio: float = 0.3, seed: int = 42):
        """
        Split engines into training and demo sets.

        Args:
            demo_ratio: Fraction of engines for demo
            seed: Random seed
        """
        np.random.seed(seed)

        all_ids = list(self._engines.keys())
        np.random.shuffle(all_ids)

        n_demo = int(len(all_ids) * demo_ratio)
        self._demo_ids = sorted(all_ids[:n_demo])
        self._train_ids = sorted(all_ids[n_demo:])

        print(f"Train engines: {len(self._train_ids)}")
        print(f"Demo engines: {len(self._demo_ids)}")

    def compute_normalization_stats(self):
        """Compute mean and std from training data."""
        if not self._train_ids:
            self.create_train_demo_split()

        # Collect all training sensor data
        all_sensors = []
        for engine_id in self._train_ids:
            engine = self._engines[engine_id]
            all_sensors.append(engine.sensors)

        all_sensors = np.vstack(all_sensors)

        self._normalization_stats = {
            "mean": all_sensors.mean(axis=0),
            "std": all_sensors.std(axis=0) + 1e-8,
        }

    def normalize_sensors(self, sensors: np.ndarray) -> np.ndarray:
        """Normalize sensor readings."""
        if self._normalization_stats is None:
            self.compute_normalization_stats()

        return (sensors - self._normalization_stats["mean"]) / self._normalization_stats["std"]

    def get_engine(self, engine_id: int) -> Optional[EngineData]:
        """Get engine by ID."""
        return self._engines.get(engine_id)

    def get_engine_at_cycle(self, engine_id: int, cycle: int) -> Optional[Dict]:
        """
        Get engine state at a specific cycle.

        Returns dict with sensors, settings, rul at that cycle.
        """
        engine = self.get_engine(engine_id)
        if engine is None:
            return None

        # Find the index for this cycle
        cycle_idx = np.searchsorted(engine.cycles, cycle)
        cycle_idx = min(cycle_idx, len(engine.cycles) - 1)

        return {
            "cycle": int(engine.cycles[cycle_idx]),
            "sensors": engine.sensors[cycle_idx],
            "settings": engine.settings[cycle_idx],
            "rul": int(engine.rul[cycle_idx]),
        }

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names (sensors to use)."""
        return FEATURE_COLUMNS

    def get_sensor_stats(self) -> Dict[str, Dict]:
        """Get statistics for each sensor."""
        if self._normalization_stats is None:
            self.compute_normalization_stats()

        stats = {}
        for i, col in enumerate([f"sensor_{j}" for j in range(1, 22)]):
            stats[col] = {
                "mean": float(self._normalization_stats["mean"][i]),
                "std": float(self._normalization_stats["std"][i]),
            }
        return stats

    def get_training_sequences(
        self,
        window_size: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training sequences for LSTM.

        Args:
            window_size: Sequence length

        Returns:
            X: (n_samples, window_size, n_features)
            y: (n_samples,) - RUL values
        """
        if not self._train_ids:
            self.create_train_demo_split()

        if self._normalization_stats is None:
            self.compute_normalization_stats()

        X_list = []
        y_list = []

        # Get feature indices (excluding dropped sensors)
        feature_indices = [i for i in range(21) if f"sensor_{i+1}" not in SENSORS_TO_DROP]

        for engine_id in self._train_ids:
            engine = self._engines[engine_id]

            # Normalize sensors
            sensors_norm = self.normalize_sensors(engine.sensors)

            # Select features
            sensors_norm = sensors_norm[:, feature_indices]

            # Create sequences
            n_cycles = len(engine.cycles)
            for i in range(window_size, n_cycles + 1):
                X_list.append(sensors_norm[i - window_size:i])
                y_list.append(engine.rul[i - 1])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        return X, y

    def get_validation_sequences(
        self,
        window_size: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get validation sequences from demo engines.

        Uses demo_ids (30% of engines) for validation.

        Args:
            window_size: Sequence length

        Returns:
            X: (n_samples, window_size, n_features)
            y: (n_samples,) - RUL values
        """
        if not self._demo_ids:
            self.create_train_demo_split()

        if self._normalization_stats is None:
            self.compute_normalization_stats()

        X_list = []
        y_list = []

        # Get feature indices (excluding dropped sensors)
        feature_indices = [i for i in range(21) if f"sensor_{i+1}" not in SENSORS_TO_DROP]

        for engine_id in self._demo_ids:
            engine = self._engines[engine_id]

            # Normalize sensors (using training stats)
            sensors_norm = self.normalize_sensors(engine.sensors)

            # Select features
            sensors_norm = sensors_norm[:, feature_indices]

            # Create sequences
            n_cycles = len(engine.cycles)
            for i in range(window_size, n_cycles + 1):
                X_list.append(sensors_norm[i - window_size:i])
                y_list.append(engine.rul[i - 1])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        return X, y

    def get_engine_sequence(
        self,
        engine_id: int,
        at_cycle: Optional[int] = None,
        window_size: int = 30,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Get a sequence for a specific engine at a specific cycle.

        Args:
            engine_id: Engine ID
            at_cycle: Cycle to get sequence ending at (None = last cycle)
            window_size: Sequence length

        Returns:
            Tuple of (sequence, true_rul) or None
        """
        engine = self.get_engine(engine_id)
        if engine is None:
            return None

        if self._normalization_stats is None:
            self.compute_normalization_stats()

        # Determine end index
        if at_cycle is None:
            end_idx = len(engine.cycles)
        else:
            end_idx = np.searchsorted(engine.cycles, at_cycle, side='right')
            end_idx = min(end_idx, len(engine.cycles))

        # Get feature indices
        feature_indices = [i for i in range(21) if f"sensor_{i+1}" not in SENSORS_TO_DROP]

        # Need at least window_size cycles
        if end_idx < window_size:
            # Pad with first cycle
            padding_needed = window_size - end_idx
            sensors_norm = self.normalize_sensors(engine.sensors[:end_idx])
            sensors_norm = sensors_norm[:, feature_indices]

            # Pad
            padding = np.tile(sensors_norm[0:1], (padding_needed, 1))
            sequence = np.vstack([padding, sensors_norm])
        else:
            start_idx = end_idx - window_size
            sensors_norm = self.normalize_sensors(engine.sensors[start_idx:end_idx])
            sequence = sensors_norm[:, feature_indices]

        true_rul = int(engine.rul[end_idx - 1]) if end_idx > 0 else int(engine.rul[0])

        return sequence.astype(np.float32), true_rul


def check_data_available(data_dir: str = "data/CMAPSSData") -> bool:
    """Check if C-MAPSS data files exist."""
    data_path = Path(data_dir)
    required_files = ["train_FD001.txt"]  # At least FD001 should exist

    if not data_path.exists():
        return False

    for f in required_files:
        if not (data_path / f).exists():
            return False

    return True


def print_download_instructions():
    """Print instructions for downloading the dataset."""
    print("""
NASA C-MAPSS Dataset Download Instructions:
==========================================

1. Download from: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

2. Or from Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

3. Extract the zip file to: data/CMAPSSData/

The directory should contain:
- train_FD001.txt, train_FD002.txt, train_FD003.txt, train_FD004.txt
- test_FD001.txt, test_FD002.txt, test_FD003.txt, test_FD004.txt
- RUL_FD001.txt, RUL_FD002.txt, RUL_FD003.txt, RUL_FD004.txt
- readme.txt
""")


if __name__ == "__main__":
    # Test data loader
    print("Testing C-MAPSS Data Loader...\n")

    if not check_data_available():
        print("Data not found!")
        print_download_instructions()
        exit(1)

    loader = CMAPSSDataLoader()
    loader.load()
    loader.create_train_demo_split()
    loader.compute_normalization_stats()

    print(f"\nFeature columns: {loader.get_feature_columns()}")
    print(f"Number of features: {len(loader.get_feature_columns())}")

    # Test getting an engine
    if loader.demo_ids:
        engine_id = loader.demo_ids[0]
        engine = loader.get_engine(engine_id)
        print(f"\nEngine {engine_id}:")
        print(f"  Dataset: {engine.dataset}")
        print(f"  Cycles: {engine.n_cycles}")
        print(f"  Max cycle: {engine.max_cycle}")
        print(f"  Final RUL: {engine.rul[-1]}")

        # Test sequence generation
        seq, rul = loader.get_engine_sequence(engine_id, window_size=30)
        print(f"  Sequence shape: {seq.shape}")
        print(f"  True RUL: {rul}")

    # Test training data
    X, y = loader.get_training_sequences(window_size=30)
    print(f"\nTraining data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  RUL range: {y.min():.0f} - {y.max():.0f}")
