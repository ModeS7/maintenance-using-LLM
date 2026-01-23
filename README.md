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

## Quick Start

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Install Ollama + model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:8b

# 3. Download NASA C-MAPSS dataset
wget "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
unzip "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" -d data/

# 4. Train model (optional - uses ground truth if skipped)
python -m src.train --epochs 50

# 5. Run demo
python -m src.app
# Open http://localhost:7860
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
