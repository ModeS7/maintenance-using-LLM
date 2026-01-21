"""
LSTM model for turbofan engine RUL (Remaining Useful Life) prediction.

Predicts the number of remaining cycles before engine failure based on
time-series sensor data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RULPredictor(nn.Module):
    """
    LSTM-based model for RUL prediction.

    Architecture:
    - LSTM layers to capture temporal patterns in sensor sequences
    - Fully connected layers for regression output
    - Outputs a single RUL value
    """

    def __init__(
        self,
        input_size: int = 14,  # Number of sensors
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Fully connected layers
        lstm_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            RUL predictions of shape (batch, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_final = h_n[-1]

        # Regression head
        rul = self.fc(h_final)

        return rul

    def predict(self, x: torch.Tensor) -> float:
        """
        Make RUL prediction for a single sequence.

        Args:
            x: Input tensor of shape (1, seq_len, input_size)

        Returns:
            Predicted RUL value (clamped to >= 0)
        """
        self.eval()
        with torch.no_grad():
            rul = self.forward(x)
            return max(0.0, rul.item())


class RULPredictorCNN(nn.Module):
    """
    CNN-LSTM hybrid model for RUL prediction.

    Uses 1D convolutions to extract local features before LSTM.
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        kernel_size: int = 5,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=64,  # Output of CNN
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            RUL predictions of shape (batch, 1)
        """
        # Transpose for CNN: (batch, input_size, seq_len)
        x = x.transpose(1, 2)

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Transpose back for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        h_final = h_n[-1]

        # Regression head
        rul = self.fc(h_final)

        return rul

    def predict(self, x: torch.Tensor) -> float:
        """Make RUL prediction for a single sequence."""
        self.eval()
        with torch.no_grad():
            rul = self.forward(x)
            return max(0.0, rul.item())


class RULPredictorAttention(nn.Module):
    """
    LSTM with attention mechanism for RUL prediction.

    Uses attention to focus on the most relevant time steps.
    """

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            RUL predictions of shape (batch, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Compute attention weights
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)

        # Regression head
        rul = self.fc(context)

        return rul

    def predict(self, x: torch.Tensor) -> float:
        """Make RUL prediction for a single sequence."""
        self.eval()
        with torch.no_grad():
            rul = self.forward(x)
            return max(0.0, rul.item())


def create_model(
    model_type: str = "lstm",
    input_size: int = 14,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a RUL prediction model.

    Args:
        model_type: "lstm", "cnn_lstm", or "attention"
        input_size: Number of input features (sensors)
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        **kwargs: Additional model arguments

    Returns:
        PyTorch model
    """
    if model_type == "lstm":
        return RULPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=kwargs.get("bidirectional", False),
        )
    elif model_type == "cnn_lstm":
        return RULPredictorCNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            kernel_size=kwargs.get("kernel_size", 5),
        )
    elif model_type == "attention":
        return RULPredictorAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing RUL Predictor models...\n")

    batch_size = 32
    seq_len = 30
    input_size = 14

    # Test LSTM model
    model = create_model("lstm", input_size=input_size)
    print(f"LSTM Model: {count_parameters(model):,} parameters")

    x = torch.randn(batch_size, seq_len, input_size)
    rul = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {rul.shape}")
    print(f"  Sample RUL predictions: {rul[:5].squeeze().tolist()}")

    # Test single prediction
    single_input = torch.randn(1, seq_len, input_size)
    rul_pred = model.predict(single_input)
    print(f"  Single prediction: {rul_pred:.1f} cycles")

    # Test CNN-LSTM model
    model_cnn = create_model("cnn_lstm", input_size=input_size)
    print(f"\nCNN-LSTM Model: {count_parameters(model_cnn):,} parameters")

    rul_cnn = model_cnn(x)
    print(f"  Output shape: {rul_cnn.shape}")

    # Test Attention model
    model_attn = create_model("attention", input_size=input_size)
    print(f"\nAttention Model: {count_parameters(model_attn):,} parameters")

    rul_attn = model_attn(x)
    print(f"  Output shape: {rul_attn.shape}")
