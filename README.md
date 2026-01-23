# Turbofan Engine Predictive Maintenance with LLM Interface

A proof-of-concept showing how conversational AI makes predictive maintenance accessible. Uses NASA C-MAPSS dataset, an LSTM model for RUL prediction, and a local LLM (Ollama) for natural language interaction.

## Key Idea

**LLM interprets, model predicts.** The LLM never makes up numbers - it calls tools to get real predictions from the LSTM model, then explains them in plain language.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Gradio UI (app.py)                          │
│   Engine Visualization │ Fleet Status │ Chat + Quick Prompts    │
│   Timeline Slider (cycles_remaining)                            │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MaintenanceAgent (llm_agent.py)                 │
│   Local LLM (Ollama) - decides which tools to call              │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Tools (tools.py)                           │
│   get_engine_status │ get_sensor_readings │ list_engines        │
│   get_fleet_summary │ get_engine_timeline                       │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RULInference (inference.py)                   │
│   LSTM model (models/rul_model.pt) → RUL predictions            │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CMAPSSDataLoader (data_loader.py)               │
│   NASA C-MAPSS dataset (709 engines, 212 for demo)              │
└─────────────────────────────────────────────────────────────────┘
```

## How the LLM Gets Data

The LLM **never sees raw data**. It uses tool-calling:

```
User: "Give me a fleet overview"
  → LLM decides to call get_fleet_summary()
  → Tool returns JSON: {healthy: 151, warning: 6, critical: 0, avg_rul: 115.9}
  → LLM interprets: "The fleet has 212 engines. 151 healthy, 6 need attention..."
```

**Why tool-calling?**
- Raw data → hallucinations
- Fine-tuning → outdated when data changes
- Tool-calling → real-time data, LLM interprets

**Realistic simulation:** Future data (`true_rul`, `max_cycle`) is hidden from the LLM - just like real deployment where you don't know when engines will fail.

## Tutorial: Getting Everything Working

### Prerequisites

- Python 3.10+
- NVIDIA GPU (recommended for training, not required for demo)
- ~2GB disk space (dataset + model + Ollama)

### Step 1: Clone and Setup Environment

```bash
git clone <repository-url>
cd maintenance-using-LLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Verify:** `python -c "import torch; print(torch.cuda.is_available())"` should print `True` if GPU available.

### Step 2: Download the Dataset

The NASA C-MAPSS dataset is ~45MB.

```bash
# Download
wget "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

# Extract to data/ folder
unzip "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" -d data/
```

**Alternative:** Download from [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and extract to `data/CMAPSSData/`.

**Verify:** You should have these files:
```
data/CMAPSSData/
├── train_FD001.txt  ... train_FD004.txt
├── test_FD001.txt   ... test_FD004.txt
├── RUL_FD001.txt    ... RUL_FD004.txt
└── readme.txt
```

### Step 3: Install Ollama (Local LLM)

Ollama runs LLMs locally. Install it:

```bash
# Linux/WSL
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows: Download from https://ollama.ai/download
```

**Start Ollama** (runs as background service):
```bash
ollama serve
# If you get "address already in use", it's already running - that's fine
```

**Pull a model** (in a new terminal):
```bash
ollama pull qwen3:8b    # Recommended: good balance of speed/quality
# OR
ollama pull llama3.1:8b # Alternative
```

**Verify:** `ollama list` should show your model.

### Step 4: Train the LSTM Model

This trains the RUL prediction model on the C-MAPSS data.

```bash
python -m src.train --epochs 50 --batch-size 256
```

**What happens:**
- Loads all 4 dataset subsets (FD001-FD004)
- Splits: 497 engines for training, 212 for demo
- Creates 30-cycle sliding windows from sensor data
- Trains LSTM to predict remaining useful life
- Saves best model to `models/rul_model.pt`

**Expected output:**
```
Loaded FD001: 100 engines
Loaded FD002: 260 engines
...
Epoch 1/50: train_loss=1234.5, val_rmse=45.2
Epoch 2/50: train_loss=892.3, val_rmse=38.1
...
Training complete. Best RMSE: 17.88
```

**Training takes:** ~5-10 minutes on GPU, ~30 minutes on CPU.

**Skip training?** The demo will use ground-truth RUL values if no model exists.

### Step 5: Run the Demo

```bash
python -m src.app
```

**Open:** http://localhost:7860

**What you should see:**
- Fleet status bar at top (shows engine health distribution)
- Cycle slider (scrub through time)
- Engine visualization (SVG turbofan diagram)
- Chat interface with quick prompt buttons

### Step 6: Try It Out

1. **Move the slider** - Watch fleet status change as you simulate time passing
2. **Click "Fleet Overview"** - LLM summarizes fleet health
3. **Select different engines** - Change dataset dropdown, then engine dropdown
4. **Click "Sensor Analysis"** - LLM identifies abnormal sensor readings
5. **Ask custom questions** - "Which engines need maintenance soon?"

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No such file: train_FD001.txt` | Check dataset is in `data/CMAPSSData/` |
| `Ollama connection refused` | Run `ollama serve` in another terminal |
| `Model not found` | Run `ollama pull qwen3:8b` |
| CUDA out of memory | Reduce batch size: `--batch-size 128` |
| Slow training | Use GPU or reduce epochs: `--epochs 20` |

### Training Options

```bash
python -m src.train --help

# Key options:
--epochs 50        # Number of training epochs
--batch-size 256   # Batch size (reduce if OOM)
--window-size 30   # Sequence length for LSTM
--hidden-size 64   # LSTM hidden dimension
--model-type lstm  # Architecture: lstm, cnn_lstm, attention
```

## Dataset

**NASA C-MAPSS** - Turbofan engine run-to-failure simulations.

| Subset | Engines | Operating Conditions | Fault Modes |
|--------|---------|---------------------|-------------|
| FD001 | 100 | 1 (Sea Level) | HPC Degradation |
| FD002 | 260 | 6 (varying) | HPC Degradation |
| FD003 | 100 | 1 (Sea Level) | HPC + Fan |
| FD004 | 249 | 6 (varying) | HPC + Fan |

**21 sensors** (temperatures, pressures, speeds), **14 used** for prediction.

## Project Structure

```
src/
├── app.py           # Gradio UI
├── llm_agent.py     # Ollama LLM with tool calling
├── tools.py         # 5 tools LLM can call
├── inference.py     # LSTM model loading & prediction
├── model.py         # LSTM architecture
├── train.py         # Training script
├── data_loader.py   # C-MAPSS data parsing
└── visualization.py # Engine SVG diagrams
```

## Severity Levels

| Level | RUL | Action |
|-------|-----|--------|
| Critical | < 30 cycles | Ground immediately |
| Warning | 30-60 | Schedule maintenance |
| Caution | 60-90 | Monitor closely |
| Healthy | >= 90 | Normal operation |

## Citation

> Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," PHM 2008.
>
> Dataset: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
