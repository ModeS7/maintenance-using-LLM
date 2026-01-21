"""
SVG visualization for turbofan engine health status.

Simplified and clear visualizations for the demo.
"""

from typing import Dict, Optional, List


# Color scheme for health status
COLORS = {
    "critical": "#ef4444",    # Red
    "warning": "#f97316",     # Orange
    "caution": "#eab308",     # Yellow
    "healthy": "#22c55e",     # Green
    "normal": "#22c55e",      # Green - normal reading
    "abnormal": "#ef4444",    # Red - abnormal reading
    "neutral": "#94a3b8",     # Gray (no data)
    "background": "#1e293b",  # Dark blue-gray
    "panel": "#334155",       # Slightly lighter panel
    "text": "#f8fafc",        # Off-white
    "text_dark": "#1e293b",   # Dark text
    "engine": "#475569",      # Engine body color
    "metal": "#64748b",       # Metallic components
    "highlight": "#3b82f6",   # Blue highlight
}


def get_severity_color(severity: str) -> str:
    """Get color for a severity level."""
    return COLORS.get(severity, COLORS["neutral"])


def get_fault_mode_info(dataset: str) -> tuple:
    """Get fault mode info based on dataset.

    Returns:
        (fault_description, affected_components)
    """
    if dataset in ["FD001", "FD002"]:
        return "HPC Degradation", ["HPC"]
    elif dataset in ["FD003", "FD004"]:
        return "HPC + Fan Degradation", ["HPC", "Fan"]
    return "Unknown", []


def generate_engine_svg(
    engine_id: int,
    dataset: str,
    current_cycle: int,
    max_cycle: int,
    predicted_rul: int,
    severity: str,
    sensors: Optional[Dict[str, Dict]] = None,
    width: int = 700,
    height: int = 400,
) -> str:
    """Generate a clear turbofan engine visualization with fault mode indication."""

    severity_color = get_severity_color(severity)
    fault_desc, affected = get_fault_mode_info(dataset)

    # Component colors based on fault mode
    fan_color = COLORS["critical"] if "Fan" in affected and severity in ["critical", "warning"] else COLORS["engine"]
    hpc_color = COLORS["critical"] if "HPC" in affected and severity in ["critical", "warning"] else COLORS["engine"]

    # Pulsing animation for affected components
    fan_anim = '<animate attributeName="opacity" values="1;0.5;1" dur="1.5s" repeatCount="indefinite"/>' if "Fan" in affected and severity == "critical" else ""
    hpc_anim = '<animate attributeName="opacity" values="1;0.5;1" dur="1.5s" repeatCount="indefinite"/>' if "HPC" in affected and severity == "critical" else ""

    # Calculate life percentage
    life_pct = (current_cycle / max_cycle * 100) if max_cycle > 0 else 0

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width: 100%; height: auto;">
  <defs>
    <linearGradient id="heatGlow" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#ef4444;stop-opacity:0.8"/>
      <stop offset="100%" style="stop-color:#f97316;stop-opacity:0.4"/>
    </linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>

  <!-- Background -->
  <rect width="{width}" height="{height}" fill="{COLORS['background']}" rx="12"/>

  <!-- Header -->
  <text x="20" y="35" font-family="system-ui, sans-serif" font-size="20" font-weight="bold" fill="{COLORS['text']}">
    Engine {engine_id}
  </text>
  <text x="20" y="55" font-family="system-ui, sans-serif" font-size="13" fill="{COLORS['text']}" opacity="0.7">
    {dataset} • Cycle {current_cycle}/{max_cycle} • {fault_desc}
  </text>

  <!-- Severity Badge -->
  <rect x="{width - 140}" y="15" width="120" height="40" rx="8" fill="{severity_color}" filter="url(#glow)"/>
  <text x="{width - 80}" y="42" text-anchor="middle" font-family="system-ui, sans-serif" font-size="16" font-weight="bold" fill="white">
    {severity.upper()}
  </text>

  <!-- RUL Display -->
  <g transform="translate({width - 200}, 75)">
    <rect x="0" y="0" width="180" height="90" rx="10" fill="{COLORS['panel']}"/>
    <text x="90" y="25" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" fill="{COLORS['text']}" opacity="0.7">REMAINING LIFE</text>
    <text x="90" y="60" text-anchor="middle" font-family="system-ui, sans-serif" font-size="32" font-weight="bold" fill="{severity_color}">
      {predicted_rul}
    </text>
    <text x="90" y="80" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="{COLORS['text']}" opacity="0.6">cycles</text>
  </g>

  <!-- Engine Schematic - Simplified Cross Section -->
  <g transform="translate(30, 90)">
    <!-- Air intake arrow -->
    <polygon points="0,100 30,85 30,115" fill="{COLORS['highlight']}" opacity="0.6"/>
    <text x="5" y="135" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}" opacity="0.5">AIR IN</text>

    <!-- Fan Section -->
    <g>
      <rect x="35" y="50" width="50" height="100" rx="5" fill="{fan_color}" stroke="{COLORS['metal']}" stroke-width="2"/>
      {fan_anim}
      <text x="60" y="105" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" font-weight="bold" fill="white">FAN</text>
      {"<text x='60' y='120' text-anchor='middle' font-family='system-ui, sans-serif' font-size='8' fill='white'>⚠ FAULT</text>" if "Fan" in affected and severity in ["critical", "warning"] else ""}
    </g>

    <!-- LPC Section -->
    <rect x="90" y="60" width="40" height="80" rx="3" fill="{COLORS['engine']}" stroke="{COLORS['metal']}" stroke-width="1"/>
    <text x="110" y="105" text-anchor="middle" font-family="system-ui, sans-serif" font-size="10" fill="white">LPC</text>

    <!-- HPC Section -->
    <g>
      <rect x="135" y="55" width="55" height="90" rx="3" fill="{hpc_color}" stroke="{COLORS['metal']}" stroke-width="2"/>
      {hpc_anim}
      <text x="162" y="105" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" font-weight="bold" fill="white">HPC</text>
      {"<text x='162' y='120' text-anchor='middle' font-family='system-ui, sans-serif' font-size='8' fill='white'>⚠ FAULT</text>" if "HPC" in affected and severity in ["critical", "warning"] else ""}
    </g>

    <!-- Combustor -->
    <rect x="195" y="65" width="35" height="70" rx="3" fill="url(#heatGlow)" stroke="#ef4444" stroke-width="1"/>
    <text x="212" y="105" text-anchor="middle" font-family="system-ui, sans-serif" font-size="8" fill="white">BURN</text>

    <!-- HPT Section -->
    <rect x="235" y="60" width="40" height="80" rx="3" fill="{COLORS['engine']}" stroke="{COLORS['metal']}" stroke-width="1"/>
    <text x="255" y="105" text-anchor="middle" font-family="system-ui, sans-serif" font-size="10" fill="white">HPT</text>

    <!-- LPT Section -->
    <rect x="280" y="55" width="45" height="90" rx="3" fill="{COLORS['engine']}" stroke="{COLORS['metal']}" stroke-width="1"/>
    <text x="302" y="105" text-anchor="middle" font-family="system-ui, sans-serif" font-size="10" fill="white">LPT</text>

    <!-- Exhaust -->
    <polygon points="330,60 370,75 370,125 330,140" fill="{COLORS['metal']}" stroke="{COLORS['metal']}" stroke-width="1"/>
    <text x="365" y="135" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}" opacity="0.5">EXHAUST</text>

    <!-- Center shaft line -->
    <line x1="35" y1="100" x2="330" y2="100" stroke="{COLORS['metal']}" stroke-width="3" opacity="0.4"/>

    <!-- Flow direction arrows -->
    <text x="200" y="170" text-anchor="middle" font-family="system-ui, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.4">
      ─────────────────────────────────▶
    </text>
    <text x="200" y="185" text-anchor="middle" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}" opacity="0.4">
      Gas Flow Direction
    </text>
  </g>

  <!-- Life Progress Bar -->
  <g transform="translate(30, {height - 60})">
    <text x="0" y="0" font-family="system-ui, sans-serif" font-size="11" fill="{COLORS['text']}" opacity="0.7">Engine Lifecycle Progress</text>
    <rect x="0" y="10" width="400" height="20" rx="4" fill="{COLORS['panel']}"/>
    <rect x="0" y="10" width="{min(400, life_pct * 4)}" height="20" rx="4" fill="{severity_color}"/>
    <text x="410" y="25" font-family="system-ui, sans-serif" font-size="11" fill="{COLORS['text']}">{life_pct:.0f}% used</text>
  </g>

  <!-- Legend -->
  <g transform="translate({width - 180}, {height - 100})">
    <text x="0" y="0" font-family="system-ui, sans-serif" font-size="10" font-weight="bold" fill="{COLORS['text']}" opacity="0.7">FAULT MODE:</text>
    <rect x="0" y="10" width="12" height="12" rx="2" fill="{COLORS['critical']}"/>
    <text x="18" y="20" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}">{fault_desc}</text>

    <text x="0" y="45" font-family="system-ui, sans-serif" font-size="10" font-weight="bold" fill="{COLORS['text']}" opacity="0.7">SEVERITY:</text>
    <rect x="0" y="55" width="12" height="12" rx="2" fill="{COLORS['healthy']}"/>
    <text x="18" y="65" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}">Healthy (≥90)</text>
    <rect x="80" y="55" width="12" height="12" rx="2" fill="{COLORS['caution']}"/>
    <text x="98" y="65" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}">Caution (60-89)</text>
    <rect x="0" y="70" width="12" height="12" rx="2" fill="{COLORS['warning']}"/>
    <text x="18" y="80" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}">Warning (30-59)</text>
    <rect x="80" y="70" width="12" height="12" rx="2" fill="{COLORS['critical']}"/>
    <text x="98" y="80" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}">Critical (&lt;30)</text>
  </g>
</svg>'''

    return svg


def generate_fleet_overview_svg(
    critical: int,
    warning: int,
    caution: int,
    healthy: int,
    total: int,
    average_rul: float,
    fleet_health_pct: float,
    width: int = 600,
    height: int = 120,
) -> str:
    """Generate a compact fleet overview bar."""

    # Calculate bar widths
    bar_width = width - 40
    c_w = int((critical / total) * bar_width) if total > 0 else 0
    w_w = int((warning / total) * bar_width) if total > 0 else 0
    ca_w = int((caution / total) * bar_width) if total > 0 else 0
    h_w = int((healthy / total) * bar_width) if total > 0 else 0

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width: 100%; height: auto;">
  <rect width="{width}" height="{height}" fill="{COLORS['background']}" rx="8"/>

  <text x="20" y="25" font-family="system-ui, sans-serif" font-size="14" font-weight="bold" fill="{COLORS['text']}">
    Fleet Status: {total} Engines
  </text>
  <text x="{width - 20}" y="25" text-anchor="end" font-family="system-ui, sans-serif" font-size="12" fill="{COLORS['text']}" opacity="0.7">
    Avg RUL: {average_rul:.0f} cycles
  </text>

  <!-- Stacked bar -->
  <g transform="translate(20, 40)">
    <rect x="0" y="0" width="{c_w}" height="30" fill="{COLORS['critical']}"/>
    <rect x="{c_w}" y="0" width="{w_w}" height="30" fill="{COLORS['warning']}"/>
    <rect x="{c_w + w_w}" y="0" width="{ca_w}" height="30" fill="{COLORS['caution']}"/>
    <rect x="{c_w + w_w + ca_w}" y="0" width="{h_w}" height="30" fill="{COLORS['healthy']}"/>
  </g>

  <!-- Labels -->
  <g transform="translate(20, 85)" font-family="system-ui, sans-serif" font-size="11">
    <rect x="0" y="0" width="10" height="10" fill="{COLORS['critical']}"/>
    <text x="14" y="9" fill="{COLORS['text']}">Critical: {critical}</text>

    <rect x="100" y="0" width="10" height="10" fill="{COLORS['warning']}"/>
    <text x="114" y="9" fill="{COLORS['text']}">Warning: {warning}</text>

    <rect x="210" y="0" width="10" height="10" fill="{COLORS['caution']}"/>
    <text x="224" y="9" fill="{COLORS['text']}">Caution: {caution}</text>

    <rect x="320" y="0" width="10" height="10" fill="{COLORS['healthy']}"/>
    <text x="334" y="9" fill="{COLORS['text']}">Healthy: {healthy}</text>

    <text x="{width - 40}" y="9" text-anchor="end" fill="{COLORS['text']}" font-weight="bold">{fleet_health_pct:.0f}% OK</text>
  </g>
</svg>'''

    return svg


def generate_timeline_svg(
    timeline: List[Dict],
    engine_id: int,
    width: int = 700,
    height: int = 180,
) -> str:
    """Generate a timeline chart showing RUL over cycles."""

    if not timeline:
        return ""

    margin = {"top": 35, "right": 20, "bottom": 35, "left": 50}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    min_cycle = min(t["cycle"] for t in timeline)
    max_cycle = max(t["cycle"] for t in timeline)
    max_rul = max(max(t["true_rul"] for t in timeline), max(t["predicted_rul"] for t in timeline), 100)
    max_rul = int((max_rul // 50 + 1) * 50)

    def x(cycle):
        return margin["left"] + ((cycle - min_cycle) / max(max_cycle - min_cycle, 1)) * plot_w

    def y(rul):
        return margin["top"] + (1 - rul / max_rul) * plot_h

    # Create path strings
    pred_path = " ".join(f"{x(t['cycle'])},{y(t['predicted_rul'])}" for t in timeline)
    true_path = " ".join(f"{x(t['cycle'])},{y(t['true_rul'])}" for t in timeline)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width: 100%; height: auto;">
  <rect width="{width}" height="{height}" fill="{COLORS['background']}" rx="8"/>

  <text x="{width//2}" y="22" text-anchor="middle" font-family="system-ui, sans-serif" font-size="13" font-weight="bold" fill="{COLORS['text']}">
    Engine {engine_id} - RUL Degradation Over Time
  </text>

  <!-- Severity zones -->
  <rect x="{margin['left']}" y="{y(max_rul)}" width="{plot_w}" height="{y(90) - y(max_rul)}" fill="{COLORS['healthy']}" opacity="0.1"/>
  <rect x="{margin['left']}" y="{y(90)}" width="{plot_w}" height="{y(60) - y(90)}" fill="{COLORS['caution']}" opacity="0.1"/>
  <rect x="{margin['left']}" y="{y(60)}" width="{plot_w}" height="{y(30) - y(60)}" fill="{COLORS['warning']}" opacity="0.1"/>
  <rect x="{margin['left']}" y="{y(30)}" width="{plot_w}" height="{y(0) - y(30)}" fill="{COLORS['critical']}" opacity="0.15"/>

  <!-- Grid lines -->
  <g stroke="{COLORS['panel']}" stroke-width="1" opacity="0.5">
    <line x1="{margin['left']}" y1="{y(90)}" x2="{width - margin['right']}" y2="{y(90)}"/>
    <line x1="{margin['left']}" y1="{y(60)}" x2="{width - margin['right']}" y2="{y(60)}"/>
    <line x1="{margin['left']}" y1="{y(30)}" x2="{width - margin['right']}" y2="{y(30)}"/>
  </g>

  <!-- Y-axis labels -->
  <g font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}" opacity="0.6">
    <text x="{margin['left'] - 5}" y="{y(max_rul) + 3}" text-anchor="end">{max_rul}</text>
    <text x="{margin['left'] - 5}" y="{y(90) + 3}" text-anchor="end">90</text>
    <text x="{margin['left'] - 5}" y="{y(60) + 3}" text-anchor="end">60</text>
    <text x="{margin['left'] - 5}" y="{y(30) + 3}" text-anchor="end">30</text>
    <text x="{margin['left'] - 5}" y="{y(0) + 3}" text-anchor="end">0</text>
  </g>

  <!-- X-axis labels -->
  <text x="{margin['left']}" y="{height - 8}" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}" opacity="0.6">{min_cycle}</text>
  <text x="{width - margin['right']}" y="{height - 8}" text-anchor="end" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}" opacity="0.6">{max_cycle}</text>
  <text x="{width//2}" y="{height - 8}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="9" fill="{COLORS['text']}" opacity="0.6">Cycle</text>

  <!-- True RUL line (dashed) -->
  <polyline points="{true_path}" fill="none" stroke="{COLORS['text']}" stroke-width="2" stroke-dasharray="4,3" opacity="0.4"/>

  <!-- Predicted RUL line -->
  <polyline points="{pred_path}" fill="none" stroke="{COLORS['highlight']}" stroke-width="2.5"/>

  <!-- Legend -->
  <g transform="translate({width - 150}, 8)" font-family="system-ui, sans-serif" font-size="9">
    <line x1="0" y1="5" x2="20" y2="5" stroke="{COLORS['highlight']}" stroke-width="2"/>
    <text x="25" y="8" fill="{COLORS['text']}">Predicted</text>
    <line x1="70" y1="5" x2="90" y2="5" stroke="{COLORS['text']}" stroke-width="2" stroke-dasharray="4,3" opacity="0.5"/>
    <text x="95" y="8" fill="{COLORS['text']}" opacity="0.7">Actual</text>
  </g>
</svg>'''

    return svg
