"""
LLM Tool functions for turbofan engine predictive maintenance.

These tools are called by the LLM to get factual information about engines.
The LLM interprets and contextualizes the data, but never makes up numbers.
"""

import json
from typing import Dict, List, Optional, Any
import numpy as np

from src.inference import RULInference, get_inference
# SENSOR_NAMES and SETTING_NAMES available in inference results


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Tool definitions for Ollama
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_engine_status",
            "description": "Get the current status and RUL (Remaining Useful Life) prediction for a specific turbofan engine. Returns predicted RUL in cycles, severity level, and lifecycle information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "engine_id": {
                        "type": "integer",
                        "description": "The unique identifier of the engine"
                    },
                    "at_cycle": {
                        "type": "integer",
                        "description": "Optional: specific cycle number to check (default: current/latest cycle)"
                    }
                },
                "required": ["engine_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sensor_readings",
            "description": "Get detailed sensor readings for an engine at a specific cycle. Includes 21 sensors (temperatures, pressures, speeds) with analysis of whether values are normal or abnormal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "engine_id": {
                        "type": "integer",
                        "description": "The unique identifier of the engine"
                    },
                    "at_cycle": {
                        "type": "integer",
                        "description": "Optional: specific cycle number (default: current/latest cycle)"
                    }
                },
                "required": ["engine_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_engines",
            "description": "List all demo engines with their current status. Can filter by severity level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity_filter": {
                        "type": "string",
                        "description": "Optional filter by severity level",
                        "enum": ["critical", "warning", "caution", "healthy"]
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fleet_summary",
            "description": "Get aggregate statistics for the entire engine fleet including count by severity level and engines needing immediate attention.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_engine_timeline",
            "description": "Get RUL predictions over an engine's lifecycle to see how health has degraded over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "engine_id": {
                        "type": "integer",
                        "description": "The unique identifier of the engine"
                    },
                    "step": {
                        "type": "integer",
                        "description": "Cycle step size for timeline (default: 10)"
                    }
                },
                "required": ["engine_id"]
            }
        }
    },
]


class ToolContext:
    """Context for tool execution."""

    def __init__(self):
        self._inference: Optional[RULInference] = None
        self._current_engine_id: Optional[int] = None
        self._current_cycle: Optional[int] = None  # eval_cycle (cycle within engine lifecycle)
        self._cycles_remaining: Optional[int] = None  # cycles until end of simulation
        self._hide_future_data: bool = True  # Treat current cycle as "now"

    @property
    def inference(self) -> RULInference:
        if self._inference is None:
            self._inference = get_inference()
        return self._inference

    def set_current_engine(self, engine_id: Optional[int] = None, cycle: Optional[int] = None, cycles_remaining: Optional[int] = None):
        """Set the currently selected engine and cycle context."""
        self._current_engine_id = engine_id
        self._current_cycle = cycle
        self._cycles_remaining = cycles_remaining


# Global tool context
_tool_context: Optional[ToolContext] = None


def get_tool_context() -> ToolContext:
    """Get or create the global tool context."""
    global _tool_context
    if _tool_context is None:
        _tool_context = ToolContext()
    return _tool_context


def get_engine_status(engine_id: int, at_cycle: Optional[int] = None) -> Dict[str, Any]:
    """
    Get the current status for an engine.

    Args:
        engine_id: The engine ID to query
        at_cycle: Optional specific cycle number (uses current slider position if not provided)

    Returns:
        Dictionary with engine status including:
        - engine_id, dataset, current_cycle
        - predicted_rul (model prediction)
        - severity, severity_description
    """
    ctx = get_tool_context()

    # Use current cycle from context if not explicitly provided
    if at_cycle is None:
        at_cycle = ctx._current_cycle

    status = ctx.inference.get_engine_status(engine_id, at_cycle)

    if status is None:
        return {"error": f"Engine {engine_id} not found"}

    # Hide future data - in real deployment you wouldn't know true RUL or when failure occurs
    if ctx._hide_future_data:
        status.pop("true_rul", None)  # Unknown in real-time
        status.pop("max_cycle", None)  # Reveals failure time
        status.pop("total_cycles", None)  # Reveals full lifecycle
        status.pop("life_remaining_pct", None)  # Based on future knowledge

    return status


def get_sensor_readings(engine_id: int, at_cycle: Optional[int] = None) -> Dict[str, Any]:
    """
    Get detailed sensor readings for an engine.

    Args:
        engine_id: The engine ID to query
        at_cycle: Optional specific cycle number (uses current slider position if not provided)

    Returns:
        Dictionary with sensor readings and analysis
    """
    ctx = get_tool_context()

    # Use current cycle from context if not explicitly provided
    if at_cycle is None:
        at_cycle = ctx._current_cycle

    result = ctx.inference.get_sensor_readings(engine_id, at_cycle)

    if result is None:
        return {"error": f"Engine {engine_id} not found"}

    # Hide future data - the 'rul' field is true RUL (future knowledge)
    if ctx._hide_future_data:
        result.pop("rul", None)

    return result


def list_engines(severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all demo engines with their current status.

    Args:
        severity_filter: Optional filter by severity ("critical", "warning", "caution", "healthy")

    Returns:
        List of engine status dictionaries
    """
    ctx = get_tool_context()
    engines = ctx.inference.list_demo_engines(severity_filter=severity_filter)

    # Hide future data from each engine entry
    if ctx._hide_future_data:
        for engine in engines:
            engine.pop("true_rul", None)
            engine.pop("total_cycles", None)

    return engines


def get_fleet_summary() -> Dict[str, Any]:
    """
    Get aggregate statistics for the fleet at the current time.

    Returns:
        Dictionary with:
        - total_engines
        - critical, warning, caution, healthy counts
        - average_rul
        - fleet_health_pct
        - immediate_attention (list of critical/warning engines)
        - dataset_breakdown
    """
    ctx = get_tool_context()

    # If we have a cycles_remaining context, use it for fleet evaluation
    if ctx._cycles_remaining is not None:
        # Get global max for reference
        global_max = 0
        for engine_id in ctx.inference.data_loader.demo_ids:
            engine = ctx.inference.data_loader.get_engine(engine_id)
            if engine and engine.max_cycle > global_max:
                global_max = engine.max_cycle

        summary = ctx.inference.get_fleet_summary_at_cycle(ctx._cycles_remaining, global_max)
    else:
        summary = ctx.inference.get_fleet_summary()

    return summary


def get_engine_timeline(engine_id: int, step: int = 10) -> Dict[str, Any]:
    """
    Get RUL predictions over an engine's lifecycle.

    Args:
        engine_id: The engine ID to query
        step: Cycle step size

    Returns:
        Dictionary with timeline of predictions
    """
    ctx = get_tool_context()
    timeline = ctx.inference.get_engine_timeline(engine_id, step)

    if timeline is None:
        return {"error": f"Engine {engine_id} not found"}

    return {"engine_id": engine_id, "timeline": timeline}


# Tool registry for the LLM agent
TOOLS = {
    "get_engine_status": get_engine_status,
    "get_sensor_readings": get_sensor_readings,
    "list_engines": list_engines,
    "get_fleet_summary": get_fleet_summary,
    "get_engine_timeline": get_engine_timeline,
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool by name with given arguments.

    Args:
        name: Tool name
        arguments: Dictionary of arguments

    Returns:
        JSON string of the result
    """
    if name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = TOOLS[name](**arguments)
        return json.dumps(result, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


def get_tool_descriptions() -> str:
    """Get human-readable descriptions of available tools."""
    descriptions = []
    for tool in TOOL_DEFINITIONS:
        func = tool["function"]
        desc = f"- **{func['name']}**: {func['description']}"
        if func["parameters"].get("properties"):
            params = ", ".join(
                f"`{p}` ({func['parameters']['properties'][p].get('type', 'any')})"
                for p in func["parameters"]["properties"]
            )
            desc += f"\n  Parameters: {params}"
        descriptions.append(desc)
    return "\n".join(descriptions)


if __name__ == "__main__":
    # Test tools
    print("Testing LLM Tools...\n")

    # Initialize context
    ctx = get_tool_context()

    print("Available tools:")
    print(get_tool_descriptions())

    print("\n" + "=" * 60)
    print("Test: get_fleet_summary()")
    result = get_fleet_summary()
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("Test: list_engines(severity_filter='critical')")
    result = list_engines(severity_filter="critical")
    print(json.dumps(result[:3] if len(result) > 3 else result, indent=2))

    # Test with a specific engine
    demo_engines = ctx.inference.data_loader.demo_ids
    if demo_engines:
        engine_id = demo_engines[0]

        print(f"\n" + "=" * 60)
        print(f"Test: get_engine_status(engine_id={engine_id})")
        result = get_engine_status(engine_id)
        print(json.dumps(result, indent=2))

        print(f"\n" + "=" * 60)
        print(f"Test: get_sensor_readings(engine_id={engine_id})")
        result = get_sensor_readings(engine_id)
        # Only show first few sensors for brevity
        if "readings" in result:
            readings = result["readings"]
            result["readings"] = dict(list(readings.items())[:5])
        print(json.dumps(result, indent=2))

        print(f"\n" + "=" * 60)
        print(f"Test: get_engine_timeline(engine_id={engine_id}, step=20)")
        result = get_engine_timeline(engine_id, step=20)
        if "timeline" in result:
            result["timeline"] = result["timeline"][:5]  # First 5 points
        print(json.dumps(result, indent=2))
