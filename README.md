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
- 5-6GB disk space (for Ollama model)

### Step 1: Clone and Setup

**Linux / macOS / WSL:**
```bash
git clone <repository-url>
cd maintenance-using-LLM

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
git clone <repository-url>
cd maintenance-using-LLM

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
git clone <repository-url>
cd maintenance-using-LLM

python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Included in repo:** Dataset (`data/CMAPSSData/`) and trained model (`models/rul_model.pt`).

### Step 2: Install Ollama (Local LLM)

| Platform | Installation |
|----------|--------------|
| **Windows** | Download installer from https://ollama.ai/download |
| **macOS** | `brew install ollama` or download from https://ollama.ai/download |
| **Linux/WSL** | `curl -fsSL https://ollama.ai/install.sh \| sh` |

After installation, pull the model (same command on all platforms):
```
ollama pull qwen3:8b
```

**Verify:** `ollama list` should show `qwen3:8b`.

> **Note (Windows):** Ollama runs automatically after installation. If you see "connection refused" errors, open the Ollama app from the Start menu.

### Step 3: Run the Demo

```
python -m src.app
```

**Open:** http://localhost:7860

### Step 4: Try It Out

1. **Move the slider** - Watch fleet status change as you simulate time passing
2. **Click "Fleet Overview"** - LLM summarizes fleet health
3. **Select different engines** - Change dataset dropdown, then engine dropdown
4. **Click "Sensor Analysis"** - LLM identifies abnormal sensor readings
5. **Ask custom questions** - "Which engines need maintenance soon?"

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Ollama connection refused` | Run `ollama serve` in another terminal |
| `Model not found` | Run `ollama pull qwen3:8b` |

### Retraining the Model (Optional)

To retrain with different parameters:

```bash
python -m src.train --epochs 50 --batch-size 256

# Key options:
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
