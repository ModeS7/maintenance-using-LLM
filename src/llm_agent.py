"""
Ollama LLM agent with tool calling for turbofan engine predictive maintenance.

Handles conversation with the LLM and executes tools when needed.
"""

import json
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass, field

import ollama
from ollama import Client

from src.tools import (
    TOOL_DEFINITIONS,
    execute_tool,
    get_tool_context,
    ToolContext,
)


# System prompt for the maintenance assistant
SYSTEM_PROMPT = """You are an expert predictive maintenance assistant for turbofan aircraft engines. You help maintenance personnel understand engine health status and make informed decisions about maintenance scheduling based on Remaining Useful Life (RUL) predictions.

CRITICAL RULES:
1. NEVER make up or guess numbers. Always use tools to get factual data.
2. When asked about engine status, RUL, or sensor readings, you MUST call the appropriate tool first.
3. Interpret and contextualize the data from tools, but don't fabricate values.
4. Be clear about what's a model prediction vs ground truth data.

RUL (REMAINING USEFUL LIFE):
- Measured in operational cycles
- RUL prediction tells how many cycles until expected engine failure
- Prediction is based on LSTM model trained on historical sensor data
- You only know the PREDICTED RUL, not the actual remaining life

SEVERITY LEVELS:
- Critical (red): RUL < 30 cycles - Immediate maintenance required, high failure risk
- Warning (orange): RUL 30-60 cycles - Schedule maintenance soon, degradation detected
- Caution (yellow): RUL 60-90 cycles - Monitor closely, early signs of wear
- Healthy (green): RUL >= 90 cycles - Normal operation, no immediate concerns

ENGINE COMPONENTS MONITORED:
- Fan: Provides most of the engine thrust
- Low-Pressure Compressor (LPC): First stage compression
- High-Pressure Compressor (HPC): Final compression before combustion
- Combustion Chamber: Where fuel-air mixture burns
- High-Pressure Turbine (HPT): Drives HPC via shaft
- Low-Pressure Turbine (LPT): Drives fan and LPC via shaft

FAULT MODES IN DATASET:
- HPC Degradation: High-Pressure Compressor efficiency loss
- Fan Degradation: Fan blade erosion or damage

KEY SENSORS (21 total, 14 used for prediction):
Temperature sensors:
- T2: Fan inlet temperature
- T24: LPC outlet temperature
- T30: HPC outlet temperature
- T50: LPT outlet temperature

Pressure sensors:
- P2: Fan inlet pressure
- P15: Bypass duct pressure
- P30: HPC outlet pressure
- Ps30: HPC static pressure

Speed sensors:
- Nf: Physical fan speed
- Nc: Physical core speed
- NRf: Corrected fan speed
- NRc: Corrected core speed

Flow sensors:
- phi: Fuel flow ratio
- BPR: Bypass ratio
- W31: HPT coolant bleed
- W32: LPT coolant bleed

OPERATIONAL SETTINGS:
- Altitude: Flight altitude affects air density
- Mach number: Aircraft speed affects inlet conditions
- Throttle resolver angle: Engine power demand

When analyzing sensor readings:
- Values beyond 2 standard deviations from the mean are flagged as abnormal
- Temperature increases often indicate efficiency degradation
- Pressure drops may indicate seal or blade erosion
- Speed variations can indicate bearing wear

When giving recommendations:
- For critical engines: Recommend immediate grounding and inspection
- For warning engines: Recommend scheduling maintenance within next few flights
- For caution engines: Continue operation with increased monitoring frequency
- For healthy engines: Continue normal operation with routine inspections

DATASET INFO:
This system uses NASA C-MAPSS data (Commercial Modular Aero-Propulsion System Simulation):
- FD001: Single operating condition, single fault mode (HPC degradation)
- FD002: Six operating conditions, single fault mode
- FD003: Single operating condition, two fault modes (HPC + Fan)
- FD004: Six operating conditions, two fault modes

Be helpful, concise, and actionable. Maintenance crews need clear guidance, not lengthy explanations."""


@dataclass
class Message:
    """Chat message."""
    role: str  # "user", "assistant", "system", or "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None


@dataclass
class ConversationState:
    """State for a conversation with the LLM."""
    messages: List[Message] = field(default_factory=list)
    model: str = "qwen3:8b"
    tool_context: Optional[ToolContext] = None

    def __post_init__(self):
        if self.tool_context is None:
            self.tool_context = get_tool_context()


class MaintenanceAgent:
    """
    LLM agent for turbofan engine predictive maintenance assistance.

    Uses Ollama for local LLM inference with tool calling.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(host=host)
        self.conversation = ConversationState(model=model)
        self._check_model()

    def _check_model(self):
        """Check if the model is available in Ollama."""
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]

            available = any(
                self.model == name or name.startswith(self.model.split(':')[0])
                for name in model_names
            )

            if not available:
                print(f"Warning: Model '{self.model}' may not be available.")
                print(f"Available models: {model_names}")
                print(f"Run: ollama pull {self.model}")
        except Exception as e:
            print(f"Warning: Could not check Ollama models: {e}")
            print("Make sure Ollama is running: ollama serve")

    def set_current_engine(self, engine_id: Optional[int] = None, cycle: Optional[int] = None, cycles_remaining: Optional[int] = None):
        """Set the currently selected engine in context."""
        self.conversation.tool_context.set_current_engine(engine_id, cycle, cycles_remaining)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation.messages = []

    def _format_messages_for_ollama(self) -> List[Dict]:
        """Format conversation history for Ollama API."""
        # Build dynamic system prompt with current context
        system_content = SYSTEM_PROMPT

        # Add current time context if available
        ctx = self.conversation.tool_context
        if ctx._current_cycle is not None:
            system_content += f"""

CURRENT CONTEXT:
- Current operational cycle: {ctx._current_cycle}
- This is the PRESENT moment - you have no knowledge of future events
- All predictions are based on sensor data up to cycle {ctx._current_cycle}
- You do NOT know when the engine will actually fail - only the predicted RUL from the model"""

        if ctx._current_engine_id is not None:
            system_content += f"\n- Currently viewing engine: {ctx._current_engine_id}"

        messages = [{"role": "system", "content": system_content}]

        for msg in self.conversation.messages:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                assistant_msg = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = msg.tool_calls
                messages.append(assistant_msg)
            elif msg.role == "tool":
                messages.append({"role": "tool", "content": msg.content})

        return messages

    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        results = []

        for call in tool_calls:
            func = call.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(name, args)

            results.append({
                "tool_call_id": call.get("id", ""),
                "name": name,
                "result": result,
            })

        return results

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.

        Handles tool calling automatically.

        Args:
            user_message: The user's message

        Returns:
            The assistant's response
        """
        self.conversation.messages.append(Message(role="user", content=user_message))
        messages = self._format_messages_for_ollama()

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as e:
            return f"Error communicating with Ollama: {e}"

        message = response.message

        # Handle tool calls
        max_iterations = 5
        iteration = 0

        while message.tool_calls and iteration < max_iterations:
            iteration += 1

            tool_results = self._execute_tool_calls(message.tool_calls)

            self.conversation.messages.append(Message(
                role="assistant",
                content=message.content or "",
                tool_calls=message.tool_calls,
            ))

            for result in tool_results:
                self.conversation.messages.append(Message(
                    role="tool",
                    content=result["result"],
                ))
                messages.append({"role": "tool", "content": result["result"]})

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": message.tool_calls,
            })

            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                )
                message = response.message
            except Exception as e:
                return f"Error during tool execution: {e}"

        final_content = message.content or "I apologize, but I couldn't generate a response."

        self.conversation.messages.append(Message(
            role="assistant",
            content=final_content,
        ))

        return final_content

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Send a message and stream the response.

        Tool calls are handled before streaming starts.
        """
        self.conversation.messages.append(Message(role="user", content=user_message))
        messages = self._format_messages_for_ollama()

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as e:
            yield f"Error: {e}"
            return

        message = response.message

        # Handle tool calls (not streamed)
        if message.tool_calls:
            max_iterations = 5
            iteration = 0

            while message.tool_calls and iteration < max_iterations:
                iteration += 1

                tool_results = self._execute_tool_calls(message.tool_calls)

                self.conversation.messages.append(Message(
                    role="assistant",
                    content=message.content or "",
                    tool_calls=message.tool_calls,
                ))

                for result in tool_results:
                    self.conversation.messages.append(Message(
                        role="tool",
                        content=result["result"],
                    ))
                    messages.append({"role": "tool", "content": result["result"]})

                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": message.tool_calls,
                })

                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                )
                message = response.message

        # Stream the final response
        messages = self._format_messages_for_ollama()

        full_response = ""
        try:
            for chunk in self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
            ):
                if chunk.message.content:
                    full_response += chunk.message.content
                    yield chunk.message.content
        except Exception as e:
            yield f"\nError during streaming: {e}"
            return

        self.conversation.messages.append(Message(
            role="assistant",
            content=full_response,
        ))


def check_ollama_available(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible."""
    try:
        client = Client(host=host)
        client.list()
        return True
    except Exception:
        return False


def list_available_models(host: str = "http://localhost:11434") -> List[str]:
    """List available Ollama models."""
    try:
        client = Client(host=host)
        models = client.list()
        return [m.model for m in models.models]
    except Exception:
        return []


if __name__ == "__main__":
    print("Testing Turbofan Maintenance Agent...\n")

    if not check_ollama_available():
        print("ERROR: Ollama is not running!")
        print("Please start Ollama with: ollama serve")
        print("Then pull a model with: ollama pull qwen3:8b")
        exit(1)

    models = list_available_models()
    print(f"Available models: {models}")

    model = "qwen3:8b" if "qwen3:8b" in models else (models[0] if models else "qwen3:8b")
    print(f"Using model: {model}\n")

    agent = MaintenanceAgent(model=model)

    test_prompts = [
        "Give me a fleet overview",
        "Which engines need immediate attention?",
    ]

    for prompt in test_prompts:
        print(f"User: {prompt}")
        print(f"Assistant: ", end="")
        response = agent.chat(prompt)
        print(response)
        print()
