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


def get_engine_options() -> List[Tuple[str, int]]:
    """Get list of demo engines for dropdown."""
    inference = get_app_inference()
    options = []

    for engine_id in inference.data_loader.demo_ids[:50]:  # Limit to 50 for performance
        engine = inference.data_loader.get_engine(engine_id)
        if engine:
            prediction = inference.predict(engine_id)
            severity = prediction.severity if prediction else "unknown"
            severity_emoji = {
                "critical": "🔴",
                "warning": "🟠",
                "caution": "🟡",
                "healthy": "🟢",
            }.get(severity, "⚪")
            label = f"{severity_emoji} Engine {engine_id} ({engine.dataset})"
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


def get_fleet_summary_html() -> str:
    """Generate fleet summary HTML."""
    inference = get_app_inference()
    summary = inference.get_fleet_summary()

    svg = generate_fleet_overview_svg(
        critical=summary["critical"],
        warning=summary["warning"],
        caution=summary["caution"],
        healthy=summary["healthy"],
        total=summary["total_engines"],
        average_rul=summary["average_rul"],
        fleet_health_pct=summary["fleet_health_pct"],
    )

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

    with gr.Blocks(
        title="Turbofan Engine Predictive Maintenance",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    ) as demo:

        gr.Markdown("""
        # Turbofan Engine Predictive Maintenance Demo

        This demo shows how a conversational AI can make predictive maintenance data accessible.
        Select an engine, navigate through its lifecycle using the timeline slider, and ask questions!

        The system predicts **Remaining Useful Life (RUL)** - the number of cycles until engine failure.
        """)

        if not data_available:
            gr.Markdown("""
            **Dataset not found!**

            Please download the NASA C-MAPSS dataset:

            1. Download from: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
            2. Extract to: `data/CMAPSSData/`

            Or use the Kaggle mirror: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
            """)
            return demo

        # Top section: Engine selection and timeline
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    engine_dropdown = gr.Dropdown(
                        choices=get_engine_options(),
                        label="Select Engine",
                        value=get_engine_options()[0][1] if get_engine_options() else None,
                        interactive=True,
                        scale=3,
                    )
                    refresh_btn = gr.Button("🔄", scale=0, min_width=50)
            with gr.Column(scale=3):
                cycle_slider = gr.Slider(
                    minimum=30,
                    maximum=200,
                    value=100,
                    step=1,
                    label="Cycle (Timeline Navigation)",
                )

        # Timeline visualization
        timeline_viz = gr.HTML(value="<p>Select an engine</p>")

        # Main content: Visualization + Chat
        with gr.Row():
            # Left: Engine visualization and info
            with gr.Column(scale=1):
                engine_viz = gr.HTML(value="<p>Select an engine to view</p>")
                engine_info = gr.Markdown("Select an engine to view details")

            # Right: Chat interface
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Maintenance Assistant", height=350)

                with gr.Row():
                    btn_fleet = gr.Button("Fleet Status", size="sm")
                    btn_engine = gr.Button("This Engine", size="sm")
                    btn_sensors = gr.Button("Sensors", size="sm")
                    btn_recommend = gr.Button("Recommend", size="sm")

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about engine health, RUL, sensors...",
                        lines=1,
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    btn_attention = gr.Button("Needs Attention?", size="sm")
                    btn_timeline = gr.Button("Timeline Analysis", size="sm")
                    btn_clear = gr.Button("Clear", size="sm", variant="secondary")

        # Fleet overview at bottom
        gr.Markdown("### Fleet Overview")
        fleet_viz = gr.HTML(
            value=get_fleet_summary_html() if data_available else "<p>Loading...</p>",
        )

        # Status bar
        with gr.Row():
            ollama_status = gr.Markdown(
                "🔴 Checking Ollama..." if not check_ollama_available()
                else f"🟢 Ollama connected | Models: {', '.join(list_available_models()[:3])}"
            )

        # Event handlers
        def on_engine_change(engine_id):
            if engine_id is None:
                return "<p>No engine selected</p>", "Select an engine", 30, 200, 100, "<p>Select an engine</p>"

            # Get cycle range
            min_cycle, max_cycle, current = get_cycle_range(engine_id)

            viz = update_visualization(engine_id, current)
            info = update_engine_info(engine_id, current)
            timeline = update_timeline(engine_id)

            return viz, info, min_cycle, max_cycle, current, timeline

        def on_cycle_change(engine_id, cycle):
            if engine_id is None:
                return "<p>No engine selected</p>", "Select an engine"

            viz = update_visualization(engine_id, int(cycle))
            info = update_engine_info(engine_id, int(cycle))

            return viz, info

        def refresh_options():
            return gr.Dropdown(choices=get_engine_options())

        def submit_message(message, history, engine_id, cycle):
            if not message.strip():
                return history, ""

            history = history + [[message, None]]
            return history, ""

        def generate_response(history, engine_id, cycle):
            if not history or history[-1][1] is not None:
                return history

            message = history[-1][0]
            history[-1][1] = ""

            for response in chat_response(message, history[:-1], engine_id, cycle):
                history[-1][1] = response
                yield history

        def send_prepared_prompt(prompt, history, engine_id):
            formatted = handle_prepared_prompt(prompt, engine_id)
            history = history + [[formatted, None]]
            return history

        def clear_chat():
            agent = get_app_agent()
            if agent:
                agent.clear_history()
            return []

        # Connect events
        engine_dropdown.change(
            on_engine_change,
            inputs=[engine_dropdown],
            outputs=[engine_viz, engine_info, cycle_slider, cycle_slider, cycle_slider, timeline_viz],
        )

        cycle_slider.change(
            on_cycle_change,
            inputs=[engine_dropdown, cycle_slider],
            outputs=[engine_viz, engine_info],
        )

        refresh_btn.click(
            refresh_options,
            outputs=[engine_dropdown],
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
        if get_engine_options():
            demo.load(
                on_engine_change,
                inputs=[engine_dropdown],
                outputs=[engine_viz, engine_info, cycle_slider, cycle_slider, cycle_slider, timeline_viz],
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
