"""
Gradio UI for Turbofan Engine Predictive Maintenance Demo.

A conversational interface for exploring engine health predictions with timeline navigation.
"""

import gradio as gr
from typing import List, Tuple, Optional, Generator

from src.inference import RULInference, get_inference
from src.llm_agent import MaintenanceAgent, check_ollama_available, list_available_models
from src.tools import get_tool_context
from src.visualization import generate_engine_svg, generate_fleet_overview_svg, generate_timeline_svg
from src.data_loader import check_data_available, print_download_instructions, SENSOR_NAMES


# Global state
_inference: Optional[RULInference] = None
_agent: Optional[MaintenanceAgent] = None


def get_app_inference() -> RULInference:
    """Get or create the inference instance."""
    global _inference
    if _inference is None:
        _inference = get_inference()
    return _inference


def get_app_agent() -> Optional[MaintenanceAgent]:
    """Get or create the LLM agent instance."""
    global _agent
    if _agent is None:
        if check_ollama_available():
            models = list_available_models()
            # Prefer qwen3 or llama3.1
            preferred = ["qwen3:8b", "qwen3", "llama3.1:8b", "llama3.1"]
            model = next((m for m in preferred if any(m in avail for avail in models)), None)
            if model is None and models:
                model = models[0]
            if model:
                _agent = MaintenanceAgent(model=model)
    return _agent


def get_dataset_options() -> List[Tuple[str, str]]:
    """Get list of datasets with fault mode info."""
    return [
        ("FD001 - HPC Degradation (1 condition)", "FD001"),
        ("FD002 - HPC Degradation (6 conditions)", "FD002"),
        ("FD003 - HPC + Fan Degradation (1 condition)", "FD003"),
        ("FD004 - HPC + Fan Degradation (6 conditions)", "FD004"),
    ]


def get_engine_options(dataset: str) -> List[Tuple[str, int]]:
    """Get list of demo engines for a specific dataset."""
    inference = get_app_inference()
    options = []

    for engine_id in inference.data_loader.demo_ids:
        engine = inference.data_loader.get_engine(engine_id)
        if engine and engine.dataset == dataset:
            prediction = inference.predict(engine_id)
            severity = prediction.severity if prediction else "unknown"
            severity_emoji = {
                "critical": "🔴",
                "warning": "🟠",
                "caution": "🟡",
                "healthy": "🟢",
            }.get(severity, "⚪")
            label = f"{severity_emoji} Engine {engine_id}"
            options.append((label, engine_id))

    return options


def update_visualization(engine_id: int, cycle: Optional[int] = None) -> str:
    """Update engine visualization based on current selection and cycle."""
    inference = get_app_inference()
    engine = inference.data_loader.get_engine(engine_id)

    if engine is None:
        return "<p>No engine selected</p>"

    # Get engine status at specified cycle
    status = inference.get_engine_status(engine_id, cycle)

    if status is None:
        return "<p>Error getting engine status</p>"

    # Get sensor readings with abnormal status
    readings = inference.get_sensor_readings(engine_id, cycle)
    sensors = {}
    if readings:
        for name, data in readings["readings"].items():
            sensors[name] = {
                "value": data["value"],
                "unit": data["unit"],
                "is_abnormal": data["is_abnormal"],
            }

    # Generate SVG
    svg = generate_engine_svg(
        engine_id=engine_id,
        dataset=status["dataset"],
        current_cycle=status["current_cycle"],
        max_cycle=status["max_cycle"],
        predicted_rul=status["predicted_rul"],
        severity=status["severity"],
        sensors=sensors,
    )

    return svg


def update_engine_info(engine_id: int, cycle: Optional[int] = None) -> str:
    """Update engine information text."""
    inference = get_app_inference()
    status = inference.get_engine_status(engine_id, cycle)

    if status is None:
        return "No engine selected"

    severity_color = {
        "critical": "red",
        "warning": "orange",
        "caution": "goldenrod",
        "healthy": "green",
    }.get(status["severity"], "gray")

    info = f"""
**Engine:** {status['engine_id']} (Dataset: {status['dataset']})

**Current Cycle:** {status['current_cycle']} / {status['max_cycle']}

**Predicted RUL:** <span style="color: {severity_color}; font-weight: bold;">{status['predicted_rul']} cycles</span>

**True RUL:** {status['true_rul']} cycles

**Severity:** <span style="color: {severity_color}; font-weight: bold;">{status['severity'].upper()}</span>

**Status:** {status['severity_description']}
"""
    return info


def update_timeline(engine_id: int) -> str:
    """Generate timeline visualization for an engine."""
    inference = get_app_inference()
    timeline = inference.get_engine_timeline(engine_id, step=10)

    if timeline is None or len(timeline) == 0:
        return "<p>No timeline data available</p>"

    return generate_timeline_svg(timeline, engine_id)


def get_cycle_range(engine_id: int) -> Tuple[int, int, int]:
    """Get the cycle range for an engine."""
    inference = get_app_inference()
    engine = inference.data_loader.get_engine(engine_id)

    if engine is None:
        return 1, 100, 50

    return 30, int(engine.max_cycle), int(engine.max_cycle)


def get_global_max_cycle() -> int:
    """Get the maximum cycle across all demo engines."""
    inference = get_app_inference()
    max_cycle = 100
    for engine_id in inference.data_loader.demo_ids:
        engine = inference.data_loader.get_engine(engine_id)
        if engine and engine.max_cycle > max_cycle:
            max_cycle = engine.max_cycle
    return int(max_cycle)


def get_fleet_summary_at_cycle(cycles_remaining: int) -> str:
    """Generate fleet summary at a given cycles remaining position.

    Args:
        cycles_remaining: Cycles until failure (0 = at failure, max = just started)
    """
    inference = get_app_inference()

    # Count severity at this position
    critical = warning = caution = healthy = 0
    total_rul = 0

    global_max = get_global_max_cycle()

    for engine_id in inference.data_loader.demo_ids:
        engine = inference.data_loader.get_engine(engine_id)
        if engine is None:
            continue

        # Calculate where this engine is at this "cycles remaining" position
        # If cycles_remaining > engine.max_cycle, engine hasn't started yet (healthy)
        if cycles_remaining >= engine.max_cycle:
            # Engine is brand new or hasn't started degrading
            healthy += 1
            total_rul += engine.max_cycle
        else:
            # Engine is at cycle = (engine.max_cycle - cycles_remaining)
            eval_cycle = engine.max_cycle - cycles_remaining
            eval_cycle = max(30, eval_cycle)  # Need at least window_size

            prediction = inference.predict(engine_id, at_cycle=eval_cycle)

            if prediction:
                total_rul += prediction.predicted_rul
                if prediction.severity == "critical":
                    critical += 1
                elif prediction.severity == "warning":
                    warning += 1
                elif prediction.severity == "caution":
                    caution += 1
                else:
                    healthy += 1

    total = critical + warning + caution + healthy
    avg_rul = total_rul / total if total > 0 else 0
    health_pct = (healthy / total * 100) if total > 0 else 0

    return generate_fleet_overview_svg(
        critical=critical,
        warning=warning,
        caution=caution,
        healthy=healthy,
        total=total,
        average_rul=avg_rul,
        fleet_health_pct=health_pct,
    )


def get_fleet_summary_html() -> str:
    """Generate fleet summary HTML at start (all engines healthy)."""
    global_max = get_global_max_cycle()
    return get_fleet_summary_at_cycle(global_max)

    return svg


def chat_response(
    message: str,
    history: List[List[str]],
    engine_id: int,
    cycle: int,
) -> Generator[str, None, None]:
    """
    Process chat message and stream response.

    Args:
        message: User message
        history: Chat history
        engine_id: Currently selected engine
        cycle: Current cycle being viewed
    """
    agent = get_app_agent()

    if agent is None:
        yield "LLM not available. Please ensure Ollama is running and a model is installed.\n\nRun:\n```\nollama serve\nollama pull qwen3:8b\n```"
        return

    # Update tool context with current engine and cycle
    if engine_id:
        agent.set_current_engine(engine_id, cycle)

    # Stream response
    response = ""
    try:
        for chunk in agent.chat_stream(message):
            response += chunk
            yield response
    except Exception as e:
        yield f"Error: {str(e)}"


def handle_prepared_prompt(prompt: str, engine_id: int) -> str:
    """Format a prepared prompt with engine context."""
    if "this engine" in prompt.lower():
        return prompt.replace("this engine", f"engine {engine_id}")
    return prompt


def create_demo():
    """Create the Gradio demo interface."""

    # Check prerequisites
    data_dir = "data/CMAPSSData"
    data_available = check_data_available(data_dir)

    # Get global max cycle for slider
    global_max = get_global_max_cycle() if data_available else 500

    with gr.Blocks(title="Turbofan Engine Predictive Maintenance") as demo:

        gr.Markdown("# Turbofan Engine Predictive Maintenance")

        if not data_available:
            gr.Markdown("""
            **Dataset not found!** Download from:
            https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
            """)
            return demo

        # CSS to flip slider only
        gr.HTML("""
        <style>
        #fleet-timeline input[type="range"] {
            transform: scaleX(-1);
        }
        </style>
        """)

        # Timeline slider - flipped visually so 481 is on left, 0 on right
        # Slider value = cycles_remaining directly
        cycle_slider = gr.Slider(
            minimum=0,
            maximum=global_max,
            value=global_max,
            step=1,
            label="Cycles Remaining (Start → Failure)",
            elem_id="fleet-timeline",
        )

        # Fleet overview - updates with slider
        fleet_viz = gr.HTML(value=get_fleet_summary_html() if data_available else "")

        # Engine selection row
        with gr.Row():
            dataset_dropdown = gr.Dropdown(
                choices=get_dataset_options(),
                label="Dataset (Fault Mode)",
                value="FD001",
                interactive=True,
            )
            engine_dropdown = gr.Dropdown(
                choices=get_engine_options("FD001"),
                label="Engine",
                value=get_engine_options("FD001")[0][1] if get_engine_options("FD001") else None,
                interactive=True,
            )

        # Main content
        with gr.Row():
            with gr.Column(scale=3):
                engine_viz = gr.HTML(value="<p>Select an engine</p>")
                timeline_viz = gr.HTML(value="")
            with gr.Column(scale=2):
                engine_info = gr.Markdown("Select an engine to view details")
                chatbot = gr.Chatbot(label="Ask the AI", height=300)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about this engine...",
                        lines=1,
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    btn_fleet = gr.Button("Fleet", size="sm")
                    btn_engine = gr.Button("This Engine", size="sm")
                    btn_sensors = gr.Button("Sensors", size="sm")
                    btn_recommend = gr.Button("Recommend", size="sm")
                    btn_clear = gr.Button("Clear", size="sm", variant="secondary")

        # Hidden buttons for compatibility
        btn_attention = gr.Button("Needs Attention?", visible=False)
        btn_timeline = gr.Button("Timeline", visible=False)

        # Event handlers
        def on_cycle_slider_change(cycles_remaining, engine_id):
            """Update fleet status and engine view when slider changes."""
            # Slider is visually flipped with scaleX(-1)
            # Left = max (start), Right = 0 (failure)
            # Slider value IS cycles_remaining directly
            cycles_remaining = int(cycles_remaining)

            # Update fleet status
            fleet_html = get_fleet_summary_at_cycle(cycles_remaining)

            # Update engine view if one is selected
            if engine_id is None:
                return fleet_html, "<p>No engine selected</p>", "Select an engine"

            engine = get_app_inference().data_loader.get_engine(engine_id)
            if engine is None:
                return fleet_html, "<p>No engine selected</p>", "Select an engine"

            # Calculate engine's cycle based on cycles_remaining
            if cycles_remaining >= engine.max_cycle:
                # Engine hasn't started yet, show at cycle 30 (start)
                eval_cycle = 30
            else:
                eval_cycle = engine.max_cycle - cycles_remaining
                eval_cycle = max(30, eval_cycle)

            viz = update_visualization(engine_id, eval_cycle)
            info = update_engine_info(engine_id, eval_cycle)

            return fleet_html, viz, info

        def on_dataset_change(dataset):
            """Update engine dropdown when dataset changes."""
            options = get_engine_options(dataset)
            first_engine = options[0][1] if options else None
            return gr.Dropdown(choices=options, value=first_engine)

        def on_engine_change(engine_id, cycles_remaining):
            if engine_id is None:
                return "<p>No engine selected</p>", "Select an engine", "<p>Select an engine</p>"

            engine = get_app_inference().data_loader.get_engine(engine_id)
            if engine is None:
                return "<p>No engine selected</p>", "Select an engine", "<p>Select an engine</p>"

            # Slider value IS cycles_remaining directly (slider is visually flipped)
            cycles_remaining = int(cycles_remaining)

            # Calculate engine's cycle based on current slider position
            if cycles_remaining >= engine.max_cycle:
                eval_cycle = 30
            else:
                eval_cycle = engine.max_cycle - cycles_remaining
                eval_cycle = max(30, eval_cycle)

            viz = update_visualization(engine_id, eval_cycle)
            info = update_engine_info(engine_id, eval_cycle)
            timeline = update_timeline(engine_id)

            return viz, info, timeline

        def extract_content(content):
            """Extract plain string from content (handles structured format)."""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Structured format: [{'text': '...', 'type': 'text'}]
                texts = []
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        texts.append(item['text'])
                    elif isinstance(item, str):
                        texts.append(item)
                return ''.join(texts)
            return str(content)

        def submit_message(message, history, engine_id, cycle):
            if not message.strip():
                return history, ""

            history = list(history) if history else []
            history.append({"role": "user", "content": message})
            return history, ""

        def generate_response(history, engine_id, cycle):
            if not history:
                return history

            history = list(history)
            last_msg = history[-1]
            if not isinstance(last_msg, dict) or last_msg.get("role") != "user":
                return history

            message = extract_content(last_msg.get("content", ""))

            # Convert to old format for chat_response function
            old_history = []
            for i in range(0, len(history) - 1, 2):
                user_content = extract_content(history[i].get("content", "")) if i < len(history) else ""
                asst_content = extract_content(history[i+1].get("content", "")) if i+1 < len(history) else ""
                old_history.append([user_content, asst_content])

            # Add assistant message placeholder
            history.append({"role": "assistant", "content": ""})

            for response in chat_response(message, old_history, engine_id, cycle):
                history[-1] = {"role": "assistant", "content": str(response)}
                yield history

        def send_prepared_prompt(prompt, history, engine_id):
            formatted = handle_prepared_prompt(prompt, engine_id)
            history = list(history) if history else []
            history.append({"role": "user", "content": formatted})
            return history

        def clear_chat():
            agent = get_app_agent()
            if agent:
                agent.clear_history()
            return []

        # Connect events
        cycle_slider.change(
            on_cycle_slider_change,
            inputs=[cycle_slider, engine_dropdown],
            outputs=[fleet_viz, engine_viz, engine_info],
        )

        dataset_dropdown.change(
            on_dataset_change,
            inputs=[dataset_dropdown],
            outputs=[engine_dropdown],
        )

        engine_dropdown.change(
            on_engine_change,
            inputs=[engine_dropdown, cycle_slider],
            outputs=[engine_viz, engine_info, timeline_viz],
        )

        # Chat submission
        msg_input.submit(
            submit_message,
            inputs=[msg_input, chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot, msg_input],
        ).then(
            generate_response,
            inputs=[chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot],
        )

        send_btn.click(
            submit_message,
            inputs=[msg_input, chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot, msg_input],
        ).then(
            generate_response,
            inputs=[chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot],
        )

        # Prepared prompts
        btn_fleet.click(
            lambda h, e: send_prepared_prompt("Give me a fleet overview", h, e),
            inputs=[chatbot, engine_dropdown],
            outputs=[chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot],
        )

        btn_attention.click(
            lambda h, e: send_prepared_prompt("Which engines need immediate attention?", h, e),
            inputs=[chatbot, engine_dropdown],
            outputs=[chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot],
        )

        btn_engine.click(
            lambda h, e: send_prepared_prompt("Tell me about this engine", h, e),
            inputs=[chatbot, engine_dropdown],
            outputs=[chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot],
        )

        btn_sensors.click(
            lambda h, e: send_prepared_prompt("Analyze the sensor readings for this engine. Which are abnormal?", h, e),
            inputs=[chatbot, engine_dropdown],
            outputs=[chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot],
        )

        btn_recommend.click(
            lambda h, e: send_prepared_prompt("What maintenance actions would you recommend for this engine?", h, e),
            inputs=[chatbot, engine_dropdown],
            outputs=[chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot],
        )

        btn_timeline.click(
            lambda h, e: send_prepared_prompt(f"Show me how engine {e}'s health has degraded over time", h, e),
            inputs=[chatbot, engine_dropdown],
            outputs=[chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, engine_dropdown, cycle_slider],
            outputs=[chatbot],
        )

        btn_clear.click(clear_chat, outputs=[chatbot])

        # Initialize visualization on load
        if get_engine_options("FD001"):
            demo.load(
                on_engine_change,
                inputs=[engine_dropdown, cycle_slider],
                outputs=[engine_viz, engine_info, timeline_viz],
            )

    return demo


def main():
    """Main entry point."""
    print("Starting Turbofan Engine Predictive Maintenance Demo...")

    # Check data
    data_dir = "data/CMAPSSData"
    if check_data_available(data_dir):
        print("Dataset found")
    else:
        print("Dataset not found!")
        print_download_instructions()

    # Check Ollama
    if check_ollama_available():
        models = list_available_models()
        print(f"Ollama available with models: {models}")
    else:
        print("WARNING: Ollama not available!")
        print("Run: ollama serve && ollama pull qwen3:8b")

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
