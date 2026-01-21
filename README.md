# Turbofan Engine Predictive Maintenance Demo with LLM Interface

A proof-of-concept demonstrating how conversational AI can make predictive maintenance data accessible to maintenance personnel. This demo uses the NASA C-MAPSS Turbofan Engine Degradation dataset with a local LLM (via Ollama) and an LSTM model to predict Remaining Useful Life (RUL).

## Features

- **RUL Prediction**: LSTM model predicts remaining useful life (cycles until failure)
- **Engine Health Visualization**: Interactive SVG diagram showing turbofan engine schematic with sensor status
- **Timeline Navigation**: Scrub through an engine's operational lifecycle to see how health degrades
- **Natural Language Interface**: Ask questions about fleet health, specific engines, or sensor data
- **Tool Calling**: LLM uses structured tools to get real data - no hallucinated numbers
- **Severity Classification**: Engines classified as critical/warning/caution/healthy based on RUL

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Gradio UI                                   │
│  ┌─────────────────────────────┬─────────────────────────────┐  │
│  │   Engine Visualization      │   Fleet Overview +          │  │
│  │   (SVG turbofan schematic)  │   Chat Interface            │  │
│  ├─────────────────────────────┤                             │  │
│  │   Timeline Slider           │   Quick Prompts             │  │
│  │   (Cycle Navigation)        │   (Fleet, Sensors, etc.)    │  │
│  └─────────────────────────────┴─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Local LLM (Ollama)                           │
│                    Model: Qwen3 or Llama 3.1                    │
│                                                                 │
│   Tools: get_engine_status, get_sensor_readings,                │
│          list_engines, get_fleet_summary, get_engine_timeline   │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   RUL Prediction Model   │    │   NASA C-MAPSS Dataset   │
│   (PyTorch LSTM)         │    │   (FD001-FD004)          │
└──────────────────────────┘    └──────────────────────────┘
```

## Dataset

This demo uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**. It contains run-to-failure simulation data from a fleet of turbofan engines under varying operating conditions and fault modes.

### Dataset Subsets

| Subset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|--------|--------------|--------------|---------------------|-------------|
| FD001 | 100 | 100 | 1 (Sea Level) | 1 (HPC Degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC Degradation) |
| FD003 | 100 | 100 | 1 (Sea Level) | 2 (HPC + Fan) |
| FD004 | 248 | 249 | 6 | 2 (HPC + Fan) |

### Sensors (21 total, 14 used for prediction)

| Sensor | Name | Description | Unit |
|--------|------|-------------|------|
| T2 | Fan inlet temp | Temperature at fan inlet | °R |
| T24 | LPC outlet temp | Temperature at LPC outlet | °R |
| T30 | HPC outlet temp | Temperature at HPC outlet | °R |
| T50 | LPT outlet temp | Temperature at LPT outlet | °R |
| P2 | Fan inlet pressure | Pressure at fan inlet | psia |
| P15 | Bypass duct pressure | Total pressure in bypass duct | psia |
| P30 | HPC outlet pressure | Total pressure at HPC outlet | psia |
| Nf | Fan speed | Physical fan speed | rpm |
| Nc | Core speed | Physical core speed | rpm |
| phi | Fuel flow ratio | Ratio of fuel flow to Ps30 | pps/psi |
| BPR | Bypass ratio | Bypass ratio | - |
| W31 | HPT coolant bleed | HPT coolant bleed mass flow | lbm/s |
| W32 | LPT coolant bleed | LPT coolant bleed mass flow | lbm/s |

### Operational Settings

| Setting | Description |
|---------|-------------|
| Setting 1 | Altitude |
| Setting 2 | Mach number |
| Setting 3 | Throttle resolver angle |

### Fault Modes

- **HPC Degradation**: High-Pressure Compressor efficiency loss
- **Fan Degradation**: Fan blade erosion or damage

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) for local LLM inference
- NVIDIA GPU recommended (RTX 3090 or similar for fast training)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd maintenance-using-LLM
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and download a model**
   ```bash
   # Install Ollama (see https://ollama.ai/)
   curl -fsSL https://ollama.ai/install.sh | sh

   # Start Ollama
   ollama serve

   # Pull a model (in another terminal)
   ollama pull qwen3:8b
   # or: ollama pull llama3.1:8b
   ```

5. **Download the NASA C-MAPSS Dataset**

   Download from one of these sources:

   **Option A: PHM Data Challenge**
   ```bash
   wget "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
   unzip "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" -d data/
   ```

   **Option B: Kaggle**
   - Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
   - Extract to `data/CMAPSSData/`

   **Expected structure:**
   ```
   data/CMAPSSData/
   ├── train_FD001.txt
   ├── train_FD002.txt
   ├── train_FD003.txt
   ├── train_FD004.txt
   ├── test_FD001.txt
   ├── test_FD002.txt
   ├── test_FD003.txt
   ├── test_FD004.txt
   ├── RUL_FD001.txt
   ├── RUL_FD002.txt
   ├── RUL_FD003.txt
   ├── RUL_FD004.txt
   └── readme.txt
   ```

## Usage

### Quick Start (Demo Only)

If you just want to run the demo with ground-truth RUL values (no model training):

```bash
python -m src.app
```

Open http://localhost:7860 in your browser.

### Full Setup (With Model Training)

1. **Train the RUL prediction model**
   ```bash
   python -m src.train --epochs 50 --batch-size 256
   ```

   This will:
   - Load all four C-MAPSS subsets (FD001-FD004)
   - Create training sequences with a 30-cycle window
   - Train an LSTM model to predict RUL
   - Save the best model to `models/rul_model.pt`

2. **Run the demo**
   ```bash
   python -m src.app
   ```

### Training Options

```bash
python -m src.train --help

Options:
  --data-dir       Path to C-MAPSS data (default: data/CMAPSSData)
  --datasets       Subsets to use (default: FD001 FD002 FD003 FD004)
  --model-type     Architecture: lstm, cnn_lstm, attention (default: lstm)
  --window-size    Sequence window size (default: 30)
  --hidden-size    LSTM hidden dimension (default: 64)
  --num-layers     Number of LSTM layers (default: 2)
  --batch-size     Batch size (default: 256)
  --epochs         Maximum epochs (default: 50)
  --lr             Learning rate (default: 0.001)
  --patience       Early stopping patience (default: 10)
  --demo-ratio     Fraction of engines for demo (default: 0.3)
  --save-path      Model save path (default: models/rul_model.pt)
```

## Demo Script

When demonstrating to stakeholders:

1. **Fleet Overview**: Click "Fleet Overview" button
   - Shows aggregate health statistics by severity
   - Displays average RUL across the fleet
   - Identifies engines requiring immediate attention

2. **Critical Engines**: Click "Needs Attention?"
   - Lists engines with critical/warning severity
   - Shows predicted RUL for prioritization

3. **Timeline Navigation**: Use the cycle slider
   - Scrub through an engine's operational history
   - Watch how RUL predictions change over time
   - See the engine's degradation timeline graph

4. **Engine Deep Dive**: Select an engine, click "About This Engine"
   - Detailed RUL prediction with true value comparison
   - Engine dataset and cycle information

5. **Sensor Analysis**: Click "Sensor Analysis"
   - Identifies sensors showing abnormal readings
   - Shows values compared to normal ranges (z-scores)

6. **Recommendations**: Click "Recommendations"
   - LLM synthesizes prediction data into actionable advice
   - Context-aware maintenance scheduling suggestions

## Project Structure

```
maintenance-using-LLM/
├── data/
│   └── CMAPSSData/
│       ├── train_FD00*.txt      # Training data (run-to-failure)
│       ├── test_FD00*.txt       # Test data (pre-failure cutoff)
│       └── RUL_FD00*.txt        # True RUL for test set
├── models/
│   └── rul_model.pt             # Trained LSTM weights
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # C-MAPSS parsing, sequence generation
│   ├── model.py                 # LSTM architecture for RUL
│   ├── train.py                 # Training script with early stopping
│   ├── inference.py             # Model loading, RUL prediction
│   ├── tools.py                 # LLM tool functions
│   ├── llm_agent.py             # Ollama integration with engine context
│   ├── visualization.py         # Turbofan SVG generation
│   └── app.py                   # Gradio UI with timeline slider
├── assets/
│   └── *.svg                    # Generated visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

## Key Design Principles

1. **LLM interprets, model predicts**: The LLM never makes up numbers. All RUL values come from tool calls to the prediction system.

2. **Time-series native**: LSTM model processes sequences of sensor readings, capturing degradation patterns over time.

3. **Piecewise linear RUL**: RUL is capped at 125 cycles (constant early in engine life, then linear decline).

4. **Honest demo**: Demo engines were never seen during training - predictions are real.

5. **Timeline exploration**: Users can explore how predictions evolve throughout an engine's lifecycle.

## Severity Levels

| Level | RUL Range | Color | Action |
|-------|-----------|-------|--------|
| Critical | < 30 cycles | Red | Immediate grounding and inspection |
| Warning | 30-60 cycles | Orange | Schedule maintenance soon |
| Caution | 60-90 cycles | Yellow | Monitor closely |
| Healthy | >= 90 cycles | Green | Normal operation |

## Sensor Analysis

Sensor readings are analyzed against training data statistics:
- **Normal**: Value within 2 standard deviations of the mean
- **Abnormal**: Value beyond 2 standard deviations (potential issue)

Common patterns indicating degradation:
- Temperature increases → Efficiency loss
- Pressure drops → Seal or blade erosion
- Speed variations → Bearing wear
- Fuel flow changes → Combustion efficiency changes

## Citation

Dataset:
> A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,"
> International Conference on Prognostics and Health Management, 2008.
>
> NASA Prognostics Data Repository: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

## License

This project is for demonstration purposes. The NASA C-MAPSS dataset is publicly available from NASA's Prognostics Data Repository.
