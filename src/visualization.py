"""
SVG visualization for turbofan engine health status.

Generates a schematic diagram with color-coded sensor status and RUL indicators.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass


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
}


# Severity descriptions
SEVERITY_DESCRIPTIONS = {
    "critical": "Immediate maintenance required",
    "warning": "Schedule maintenance soon",
    "caution": "Monitor closely",
    "healthy": "Normal operation",
}


def get_severity_color(severity: str) -> str:
    """Get color for a severity level."""
    return COLORS.get(severity, COLORS["neutral"])


def get_sensor_color(is_abnormal: bool) -> str:
    """Get color for a sensor status."""
    return COLORS["abnormal"] if is_abnormal else COLORS["normal"]


def generate_engine_svg(
    engine_id: int,
    dataset: str,
    current_cycle: int,
    max_cycle: int,
    predicted_rul: int,
    severity: str,
    sensors: Optional[Dict[str, Dict]] = None,
    width: int = 800,
    height: int = 500,
) -> str:
    """
    Generate an SVG visualization of a turbofan engine with health overlay.

    Args:
        engine_id: Engine identifier
        dataset: Dataset (FD001, FD002, etc.)
        current_cycle: Current operational cycle
        max_cycle: Maximum cycle in the data
        predicted_rul: Predicted remaining useful life
        severity: Severity level
        sensors: Optional dict mapping sensor names to readings
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        SVG string
    """
    severity_color = get_severity_color(severity)
    severity_desc = SEVERITY_DESCRIPTIONS.get(severity, "Unknown")

    # RUL progress bar (max 125)
    rul_pct = min(100, (predicted_rul / 125) * 100)
    life_used_pct = 100 - rul_pct

    # Key sensors to display (if available)
    def sensor_display(name: str) -> tuple:
        if sensors and name in sensors:
            s = sensors[name]
            color = get_sensor_color(s.get("is_abnormal", False))
            value = f"{s['value']:.1f}"
            unit = s.get("unit", "")
            return value, unit, color
        return "N/A", "", COLORS["neutral"]

    # Get key sensor displays
    t30 = sensor_display("T30 (HPC outlet temp)")
    p30 = sensor_display("P30 (HPC outlet pressure)")
    nf = sensor_display("Nf (Fan speed)")
    nc = sensor_display("Nc (Core speed)")
    phi = sensor_display("phi (Fuel flow ratio)")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width: 100%; height: auto; max-width: {width}px;">
  <defs>
    <linearGradient id="engineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#475569;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#64748b;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#475569;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="fanGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#94a3b8;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#64748b;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="heatGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#ef4444;stop-opacity:0.6" />
      <stop offset="100%" style="stop-color:#f97316;stop-opacity:0.3" />
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="shadow">
      <feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.3"/>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="{width}" height="{height}" fill="{COLORS['background']}" rx="8"/>

  <!-- Title Bar -->
  <rect x="0" y="0" width="{width}" height="60" fill="{COLORS['panel']}" rx="8"/>
  <text x="20" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="{COLORS['text']}">
    Engine {engine_id} - Health Monitor
  </text>
  <text x="20" y="48" font-family="Arial, sans-serif" font-size="12" fill="{COLORS['text']}" opacity="0.7">
    Dataset: {dataset} | Cycle: {current_cycle} / {max_cycle}
  </text>

  <!-- Severity Status Badge -->
  <g transform="translate(620, 12)">
    <rect x="0" y="0" width="160" height="36" fill="{severity_color}" rx="6" filter="url(#glow)"/>
    <text x="80" y="24" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{COLORS['text_dark']}">
      {severity.upper()}
    </text>
  </g>

  <!-- Turbofan Engine Schematic -->
  <g transform="translate(30, 80)">
    <!-- Engine nacelle outline -->
    <ellipse cx="200" cy="120" rx="180" ry="95" fill="none" stroke="{COLORS['metal']}" stroke-width="3"/>

    <!-- Fan section -->
    <g transform="translate(30, 40)">
      <ellipse cx="0" cy="80" rx="15" ry="80" fill="url(#fanGradient)" stroke="{COLORS['metal']}" stroke-width="2"/>
      <!-- Fan blades -->
      <line x1="0" y1="10" x2="0" y2="150" stroke="{COLORS['metal']}" stroke-width="2"/>
      <line x1="-12" y1="30" x2="12" y2="130" stroke="{COLORS['metal']}" stroke-width="1.5"/>
      <line x1="-12" y1="130" x2="12" y2="30" stroke="{COLORS['metal']}" stroke-width="1.5"/>
      <text x="0" y="180" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.7">Fan</text>
    </g>

    <!-- LPC section -->
    <rect x="55" y="50" width="40" height="140" fill="{COLORS['engine']}" stroke="{COLORS['metal']}" stroke-width="1"/>
    <text x="75" y="210" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.7">LPC</text>

    <!-- HPC section -->
    <rect x="100" y="60" width="60" height="120" fill="{COLORS['engine']}" stroke="{COLORS['metal']}" stroke-width="1"/>
    <text x="130" y="200" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.7">HPC</text>

    <!-- Combustion chamber (with heat glow) -->
    <rect x="165" y="70" width="50" height="100" fill="url(#heatGradient)" stroke="#ef4444" stroke-width="2"/>
    <text x="190" y="190" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.7">Combustor</text>

    <!-- HPT section -->
    <rect x="220" y="65" width="40" height="110" fill="{COLORS['engine']}" stroke="{COLORS['metal']}" stroke-width="1"/>
    <text x="240" y="195" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.7">HPT</text>

    <!-- LPT section -->
    <rect x="265" y="55" width="50" height="130" fill="{COLORS['engine']}" stroke="{COLORS['metal']}" stroke-width="1"/>
    <text x="290" y="205" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.7">LPT</text>

    <!-- Exhaust nozzle -->
    <polygon points="320,70 360,90 360,150 320,170" fill="{COLORS['metal']}" stroke="{COLORS['metal']}" stroke-width="1"/>
    <text x="340" y="190" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.7">Exhaust</text>

    <!-- Center shaft -->
    <rect x="30" y="115" width="290" height="10" fill="{COLORS['metal']}" opacity="0.5"/>

    <!-- Bypass duct (upper and lower) -->
    <path d="M 50,35 L 320,60" stroke="{COLORS['metal']}" stroke-width="1" fill="none" opacity="0.5"/>
    <path d="M 50,205 L 320,180" stroke="{COLORS['metal']}" stroke-width="1" fill="none" opacity="0.5"/>
  </g>

  <!-- RUL Panel -->
  <g transform="translate(450, 80)">
    <rect x="0" y="0" width="320" height="130" fill="{COLORS['panel']}" rx="8"/>
    <text x="20" y="25" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{COLORS['text']}">
      Remaining Useful Life
    </text>

    <!-- RUL value -->
    <text x="160" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="36" font-weight="bold" fill="{severity_color}">
      {predicted_rul}
    </text>
    <text x="160" y="90" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{COLORS['text']}" opacity="0.7">
      cycles remaining
    </text>

    <!-- Progress bar (life used) -->
    <text x="20" y="115" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.7">Life Used</text>
    <rect x="80" y="105" width="200" height="12" fill="{COLORS['background']}" rx="3"/>
    <rect x="80" y="105" width="{min(200, life_used_pct * 2)}" height="12" fill="{severity_color}" rx="3"/>
    <text x="285" y="115" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">{life_used_pct:.0f}%</text>
  </g>

  <!-- Key Sensors Panel -->
  <g transform="translate(450, 220)">
    <rect x="0" y="0" width="320" height="160" fill="{COLORS['panel']}" rx="8"/>
    <text x="20" y="25" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{COLORS['text']}">
      Key Sensor Readings
    </text>

    <!-- T30 - HPC outlet temp -->
    <g transform="translate(20, 40)">
      <circle cx="8" cy="8" r="5" fill="{t30[2]}"/>
      <text x="22" y="12" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">T30 (HPC Temp)</text>
      <text x="280" y="12" text-anchor="end" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="{COLORS['text']}">{t30[0]} {t30[1]}</text>
    </g>

    <!-- P30 - HPC outlet pressure -->
    <g transform="translate(20, 65)">
      <circle cx="8" cy="8" r="5" fill="{p30[2]}"/>
      <text x="22" y="12" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">P30 (HPC Pressure)</text>
      <text x="280" y="12" text-anchor="end" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="{COLORS['text']}">{p30[0]} {p30[1]}</text>
    </g>

    <!-- Nf - Fan speed -->
    <g transform="translate(20, 90)">
      <circle cx="8" cy="8" r="5" fill="{nf[2]}"/>
      <text x="22" y="12" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">Nf (Fan Speed)</text>
      <text x="280" y="12" text-anchor="end" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="{COLORS['text']}">{nf[0]} {nf[1]}</text>
    </g>

    <!-- Nc - Core speed -->
    <g transform="translate(20, 115)">
      <circle cx="8" cy="8" r="5" fill="{nc[2]}"/>
      <text x="22" y="12" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">Nc (Core Speed)</text>
      <text x="280" y="12" text-anchor="end" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="{COLORS['text']}">{nc[0]} {nc[1]}</text>
    </g>

    <!-- phi - Fuel flow -->
    <g transform="translate(20, 140)">
      <circle cx="8" cy="8" r="5" fill="{phi[2]}"/>
      <text x="22" y="12" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">phi (Fuel Flow)</text>
      <text x="280" y="12" text-anchor="end" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="{COLORS['text']}">{phi[0]} {phi[1]}</text>
    </g>
  </g>

  <!-- Status Description Panel -->
  <g transform="translate(450, 390)">
    <rect x="0" y="0" width="320" height="95" fill="{severity_color}" opacity="0.2" rx="8"/>
    <rect x="0" y="0" width="320" height="95" fill="none" stroke="{severity_color}" stroke-width="2" rx="8"/>
    <text x="160" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{severity_color}">
      Status: {severity.upper()}
    </text>
    <text x="160" y="55" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{COLORS['text']}">
      {severity_desc}
    </text>
    <text x="160" y="80" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}" opacity="0.7">
      Predicted RUL: {predicted_rul} cycles
    </text>
  </g>

  <!-- Legend -->
  <g transform="translate(30, 320)">
    <rect x="0" y="0" width="400" height="165" fill="{COLORS['panel']}" rx="8"/>
    <text x="20" y="25" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="{COLORS['text']}">
      Severity Levels
    </text>

    <g transform="translate(20, 45)">
      <circle cx="8" cy="8" r="6" fill="{COLORS['critical']}"/>
      <text x="22" y="12" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">Critical - RUL &lt; 30 cycles - Immediate maintenance</text>
    </g>

    <g transform="translate(20, 70)">
      <circle cx="8" cy="8" r="6" fill="{COLORS['warning']}"/>
      <text x="22" y="12" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">Warning - RUL 30-60 cycles - Schedule maintenance</text>
    </g>

    <g transform="translate(20, 95)">
      <circle cx="8" cy="8" r="6" fill="{COLORS['caution']}"/>
      <text x="22" y="12" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">Caution - RUL 60-90 cycles - Monitor closely</text>
    </g>

    <g transform="translate(20, 120)">
      <circle cx="8" cy="8" r="6" fill="{COLORS['healthy']}"/>
      <text x="22" y="12" font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}">Healthy - RUL &gt;= 90 cycles - Normal operation</text>
    </g>

    <text x="20" y="155" font-family="Arial, sans-serif" font-size="9" fill="{COLORS['text']}" opacity="0.5">
      NASA C-MAPSS Turbofan Engine Degradation Dataset
    </text>
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
    width: int = 400,
    height: int = 320,
) -> str:
    """
    Generate a fleet health overview visualization.

    Args:
        critical: Count of critical engines
        warning: Count of warning engines
        caution: Count of caution engines
        healthy: Count of healthy engines
        total: Total engine count
        average_rul: Average RUL across fleet
        fleet_health_pct: Percentage of healthy engines
        width: SVG width
        height: SVG height

    Returns:
        SVG string
    """
    # Calculate bar widths
    max_bar_width = width - 140
    c_width = int((critical / total) * max_bar_width) if total > 0 else 0
    w_width = int((warning / total) * max_bar_width) if total > 0 else 0
    ca_width = int((caution / total) * max_bar_width) if total > 0 else 0
    h_width = int((healthy / total) * max_bar_width) if total > 0 else 0

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width: 100%; height: auto; max-width: {width}px;">
  <rect width="{width}" height="{height}" fill="{COLORS['background']}" rx="8"/>

  <text x="20" y="30" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="{COLORS['text']}">
    Fleet Health Overview
  </text>
  <text x="20" y="50" font-family="Arial, sans-serif" font-size="12" fill="{COLORS['text']}" opacity="0.7">
    {total} engines monitored
  </text>

  <!-- Critical bar -->
  <g transform="translate(20, 70)">
    <text x="0" y="12" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">Critical</text>
    <rect x="70" y="0" width="{c_width}" height="18" fill="{COLORS['critical']}" rx="2"/>
    <text x="{80 + c_width}" y="13" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">{critical}</text>
  </g>

  <!-- Warning bar -->
  <g transform="translate(20, 95)">
    <text x="0" y="12" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">Warning</text>
    <rect x="70" y="0" width="{w_width}" height="18" fill="{COLORS['warning']}" rx="2"/>
    <text x="{80 + w_width}" y="13" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">{warning}</text>
  </g>

  <!-- Caution bar -->
  <g transform="translate(20, 120)">
    <text x="0" y="12" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">Caution</text>
    <rect x="70" y="0" width="{ca_width}" height="18" fill="{COLORS['caution']}" rx="2"/>
    <text x="{80 + ca_width}" y="13" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">{caution}</text>
  </g>

  <!-- Healthy bar -->
  <g transform="translate(20, 145)">
    <text x="0" y="12" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">Healthy</text>
    <rect x="70" y="0" width="{h_width}" height="18" fill="{COLORS['healthy']}" rx="2"/>
    <text x="{80 + h_width}" y="13" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">{healthy}</text>
  </g>

  <!-- Summary stats -->
  <g transform="translate(20, 185)">
    <rect x="0" y="0" width="{width - 40}" height="120" fill="{COLORS['panel']}" rx="6"/>

    <text x="15" y="25" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">
      Fleet Health Score
    </text>
    <text x="{width - 60}" y="25" text-anchor="end" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{COLORS['healthy']}">
      {fleet_health_pct:.0f}%
    </text>

    <line x1="15" y1="40" x2="{width - 55}" y2="40" stroke="{COLORS['background']}" stroke-width="1"/>

    <text x="15" y="58" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">
      Average RUL
    </text>
    <text x="{width - 60}" y="58" text-anchor="end" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{COLORS['text']}">
      {average_rul:.0f} cycles
    </text>

    <line x1="15" y1="73" x2="{width - 55}" y2="73" stroke="{COLORS['background']}" stroke-width="1"/>

    <text x="15" y="91" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">
      Need Attention
    </text>
    <text x="{width - 60}" y="91" text-anchor="end" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{COLORS['critical'] if (critical + warning) > 0 else COLORS['healthy']}">
      {critical + warning}
    </text>

    <text x="15" y="110" font-family="Arial, sans-serif" font-size="9" fill="{COLORS['text']}" opacity="0.7">
      {critical} critical + {warning} warning engines require maintenance
    </text>
  </g>
</svg>'''

    return svg


def generate_timeline_svg(
    timeline: List[Dict],
    engine_id: int,
    width: int = 600,
    height: int = 200,
) -> str:
    """
    Generate a timeline visualization of RUL over cycles.

    Args:
        timeline: List of {"cycle": int, "predicted_rul": int, "true_rul": int, "severity": str}
        engine_id: Engine identifier
        width: SVG width
        height: SVG height

    Returns:
        SVG string
    """
    if not timeline:
        return ""

    # Calculate scales
    margin = {"top": 40, "right": 30, "bottom": 40, "left": 50}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    min_cycle = min(t["cycle"] for t in timeline)
    max_cycle = max(t["cycle"] for t in timeline)
    max_rul = 125

    def x_scale(cycle):
        return margin["left"] + ((cycle - min_cycle) / (max_cycle - min_cycle + 1)) * plot_width

    def y_scale(rul):
        return margin["top"] + (1 - rul / max_rul) * plot_height

    # Generate path for predicted RUL
    pred_points = " ".join(f"{x_scale(t['cycle'])},{y_scale(t['predicted_rul'])}" for t in timeline)

    # Generate path for true RUL
    true_points = " ".join(f"{x_scale(t['cycle'])},{y_scale(t['true_rul'])}" for t in timeline)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width: 100%; height: auto; max-width: {width}px;">
  <rect width="{width}" height="{height}" fill="{COLORS['background']}" rx="8"/>

  <text x="{width//2}" y="25" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{COLORS['text']}">
    Engine {engine_id} - RUL Over Time
  </text>

  <!-- Grid lines -->
  <g stroke="{COLORS['panel']}" stroke-width="1">
    <line x1="{margin['left']}" y1="{y_scale(125)}" x2="{width - margin['right']}" y2="{y_scale(125)}"/>
    <line x1="{margin['left']}" y1="{y_scale(90)}" x2="{width - margin['right']}" y2="{y_scale(90)}"/>
    <line x1="{margin['left']}" y1="{y_scale(60)}" x2="{width - margin['right']}" y2="{y_scale(60)}"/>
    <line x1="{margin['left']}" y1="{y_scale(30)}" x2="{width - margin['right']}" y2="{y_scale(30)}"/>
    <line x1="{margin['left']}" y1="{y_scale(0)}" x2="{width - margin['right']}" y2="{y_scale(0)}"/>
  </g>

  <!-- Y-axis labels -->
  <g font-family="Arial, sans-serif" font-size="10" fill="{COLORS['text']}" opacity="0.7">
    <text x="{margin['left'] - 5}" y="{y_scale(125) + 4}" text-anchor="end">125</text>
    <text x="{margin['left'] - 5}" y="{y_scale(90) + 4}" text-anchor="end">90</text>
    <text x="{margin['left'] - 5}" y="{y_scale(60) + 4}" text-anchor="end">60</text>
    <text x="{margin['left'] - 5}" y="{y_scale(30) + 4}" text-anchor="end">30</text>
    <text x="{margin['left'] - 5}" y="{y_scale(0) + 4}" text-anchor="end">0</text>
  </g>

  <!-- Severity zones -->
  <rect x="{margin['left']}" y="{y_scale(125)}" width="{plot_width}" height="{y_scale(90) - y_scale(125)}" fill="{COLORS['healthy']}" opacity="0.1"/>
  <rect x="{margin['left']}" y="{y_scale(90)}" width="{plot_width}" height="{y_scale(60) - y_scale(90)}" fill="{COLORS['caution']}" opacity="0.1"/>
  <rect x="{margin['left']}" y="{y_scale(60)}" width="{plot_width}" height="{y_scale(30) - y_scale(60)}" fill="{COLORS['warning']}" opacity="0.1"/>
  <rect x="{margin['left']}" y="{y_scale(30)}" width="{plot_width}" height="{y_scale(0) - y_scale(30)}" fill="{COLORS['critical']}" opacity="0.1"/>

  <!-- True RUL line -->
  <polyline points="{true_points}" fill="none" stroke="{COLORS['text']}" stroke-width="2" stroke-dasharray="5,3" opacity="0.5"/>

  <!-- Predicted RUL line -->
  <polyline points="{pred_points}" fill="none" stroke="{COLORS['healthy']}" stroke-width="2"/>

  <!-- X-axis label -->
  <text x="{width//2}" y="{height - 10}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">
    Operational Cycle
  </text>

  <!-- Y-axis label -->
  <text x="15" y="{height//2}" transform="rotate(-90, 15, {height//2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{COLORS['text']}">
    RUL (cycles)
  </text>

  <!-- Legend -->
  <g transform="translate({width - 130}, {margin['top']})">
    <line x1="0" y1="0" x2="20" y2="0" stroke="{COLORS['healthy']}" stroke-width="2"/>
    <text x="25" y="4" font-family="Arial, sans-serif" font-size="9" fill="{COLORS['text']}">Predicted</text>
    <line x1="0" y1="15" x2="20" y2="15" stroke="{COLORS['text']}" stroke-width="2" stroke-dasharray="5,3" opacity="0.5"/>
    <text x="25" y="19" font-family="Arial, sans-serif" font-size="9" fill="{COLORS['text']}">True</text>
  </g>
</svg>'''

    return svg


if __name__ == "__main__":
    # Test visualization
    test_sensors = {
        "T30 (HPC outlet temp)": {"value": 1580.5, "unit": "°R", "is_abnormal": False},
        "P30 (HPC outlet pressure)": {"value": 552.3, "unit": "psia", "is_abnormal": False},
        "Nf (Fan speed)": {"value": 2388.1, "unit": "rpm", "is_abnormal": False},
        "Nc (Core speed)": {"value": 9045.2, "unit": "rpm", "is_abnormal": True},
        "phi (Fuel flow ratio)": {"value": 521.8, "unit": "pps/psi", "is_abnormal": False},
    }

    svg = generate_engine_svg(
        engine_id=42,
        dataset="FD001",
        current_cycle=150,
        max_cycle=192,
        predicted_rul=42,
        severity="warning",
        sensors=test_sensors,
    )

    # Save test SVG
    from pathlib import Path
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    with open(assets_dir / "test_engine.svg", "w") as f:
        f.write(svg)
    print("Saved test_engine.svg to assets/")

    # Test fleet overview
    fleet_svg = generate_fleet_overview_svg(
        critical=5,
        warning=12,
        caution=25,
        healthy=170,
        total=212,
        average_rul=78.5,
        fleet_health_pct=80.2,
    )

    with open(assets_dir / "test_fleet.svg", "w") as f:
        f.write(fleet_svg)
    print("Saved test_fleet.svg to assets/")

    # Test timeline
    timeline = [
        {"cycle": 30, "predicted_rul": 120, "true_rul": 125, "severity": "healthy"},
        {"cycle": 50, "predicted_rul": 105, "true_rul": 105, "severity": "healthy"},
        {"cycle": 70, "predicted_rul": 88, "true_rul": 85, "severity": "caution"},
        {"cycle": 90, "predicted_rul": 65, "true_rul": 65, "severity": "caution"},
        {"cycle": 110, "predicted_rul": 48, "true_rul": 45, "severity": "warning"},
        {"cycle": 130, "predicted_rul": 28, "true_rul": 25, "severity": "critical"},
        {"cycle": 150, "predicted_rul": 8, "true_rul": 5, "severity": "critical"},
    ]

    timeline_svg = generate_timeline_svg(timeline, engine_id=42)
    with open(assets_dir / "test_timeline.svg", "w") as f:
        f.write(timeline_svg)
    print("Saved test_timeline.svg to assets/")
