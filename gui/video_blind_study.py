#!/usr/bin/env python3
"""
Video-Based Blind Study GUI for VFI+SR Model Comparison

Modes:
- Learn: Single video with labels
- Comparison: Side-by-side control vs method (auto-play, slow-mo, muted)
- Quiz: Guess the method

Features:
- Dark/Light theme toggle
- Collapsible sidebar
- Show Details overlay
- Performance stats
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict

from flask import Flask, render_template_string, jsonify, request, Response

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)

state = {
    'video_dir': None,
    'metadata': None,
    'models': [],
    'clips': [],
}

# Processing state for re-processing
processing_state = {
    'is_processing': False,
    'progress': 0,
    'status': '',
    'error': None
}


def run_benchmark():
    """Run benchmark_proper.py in background"""
    global processing_state
    processing_state['is_processing'] = True
    processing_state['progress'] = 0
    processing_state['status'] = 'Starting...'
    processing_state['error'] = None

    try:
        script_path = Path(__file__).parent / 'benchmark_proper.py'
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        methods_count = len(state.get('models', [])) or 16  # Dynamic based on loaded models
        current_method = 0

        for line in process.stdout:
            line = line.strip()
            if line.startswith('[') and '] Processing' in line:
                current_method += 1
                method = line.split(']')[0][1:]
                processing_state['status'] = f'Processing {method}...'
                processing_state['progress'] = int((current_method / methods_count) * 80)
            elif 'frames in' in line:
                processing_state['progress'] = int((current_method / methods_count) * 80) + 10

        process.wait()

        if process.returncode == 0:
            processing_state['progress'] = 100
            processing_state['status'] = 'Complete!'
            # Reload metadata
            state['metadata'] = load_video_metadata(state['video_dir'])
            state['clips'] = state['metadata'].get('clips', [])
            state['models'] = state['metadata'].get('models', [])
        else:
            processing_state['error'] = 'Benchmark failed'
            processing_state['status'] = 'Error'

    except Exception as e:
        processing_state['error'] = str(e)
        processing_state['status'] = 'Error'
    finally:
        processing_state['is_processing'] = False


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VFI+SR Blind Study</title>
    <style>
        :root {
            --bg-primary: #121212;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #252525;
            --border: #333;
            --text-primary: #fff;
            --text-secondary: #888;
            --accent: #4fc3f7;
            --success: #81c784;
            --warning: #ff9800;
            --error: #e57373;
        }
        .light-mode {
            --bg-primary: #f5f5f5;
            --bg-secondary: #fff;
            --bg-tertiary: #e8e8e8;
            --border: #ddd;
            --text-primary: #222;
            --text-secondary: #666;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            transition: background 0.3s, color 0.3s;
        }

        /* Header */
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header-left { display: flex; align-items: center; gap: 16px; }
        .header h1 { font-size: 1.2em; font-weight: 600; }
        .header-controls { display: flex; gap: 12px; align-items: center; }
        .header-stats { display: flex; gap: 24px; font-size: 0.9em; }
        .header-stats .stat { color: var(--text-secondary); }
        .header-stats .stat strong { color: var(--accent); }
        .icon-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            width: 36px; height: 36px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1em;
        }
        .icon-btn:hover { background: var(--border); }

        /* Main layout */
        .main { display: flex; height: calc(100vh - 52px); }

        /* Sidebar */
        .sidebar {
            width: 280px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            padding: 16px;
            overflow-y: auto;
            transition: width 0.3s, padding 0.3s;
        }
        .sidebar.collapsed { width: 0; padding: 0; overflow: hidden; }
        .sidebar h3 {
            font-size: 0.85em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .model-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 24px; }
        .model-item {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .model-item:hover { border-color: var(--accent); }
        .model-item.active { background: #1e3a5f; border-color: var(--accent); }
        .model-item.control { border-left: 3px solid var(--success); }
        .model-item.innovative { border-left: 3px solid var(--warning); }
        .model-item .name { font-weight: 600; font-size: 0.95em; margin-bottom: 4px; }
        .model-item .details { font-size: 0.8em; color: var(--text-secondary); }

        /* Phase toggle */
        .phase-toggle {
            display: flex;
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 4px;
            margin-bottom: 16px;
        }
        .phase-btn {
            flex: 1;
            padding: 10px 6px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 6px;
            font-size: 0.8em;
            font-weight: 500;
            transition: all 0.2s;
        }
        .phase-btn.active { background: var(--accent); color: #000; }

        /* Video area */
        .video-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0a0a0a;
        }

        /* Single video view */
        .single-view {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        .single-view video {
            max-width: 100%;
            max-height: 100%;
            transition: transform 0.5s ease;
        }

        /* Comparison view (side-by-side) */
        .comparison-view {
            flex: 1;
            display: none;
            gap: 4px;
            padding: 8px;
        }
        .comparison-view.active { display: flex; }
        .video-panel {
            flex: 1;
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }
        .video-panel video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .panel-label {
            position: absolute;
            top: 12px;
            left: 12px;
            background: rgba(0,0,0,0.8);
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9em;
        }
        .panel-label.control { color: var(--success); }
        .panel-label.method { color: var(--accent); }

        /* Video label */
        .video-label {
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(0,0,0,0.8);
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
        }
        .video-label.control { color: var(--success); }
        .video-label.method { color: var(--accent); }

        /* Details overlay */
        .details-overlay {
            position: absolute;
            bottom: 16px;
            left: 16px;
            right: 16px;
            background: rgba(0,0,0,0.85);
            padding: 16px;
            border-radius: 8px;
            display: none;
            font-size: 0.85em;
        }
        .details-overlay.show { display: block; }
        .details-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }
        .detail-item span:first-child { color: var(--text-secondary); }
        .detail-item span:last-child { color: var(--accent); font-weight: 600; }

        /* Controls bar */
        .controls-bar {
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            padding: 16px 24px;
        }
        .controls-row {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }
        .controls-row:last-child { margin-bottom: 0; }

        button {
            padding: 10px 20px;
            border: 1px solid var(--border);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
        }
        button:hover { background: var(--border); }
        button.primary { background: var(--accent); border-color: var(--accent); color: #000; }
        button.primary:hover { opacity: 0.9; }

        /* Quiz buttons */
        .quiz-buttons { display: none; gap: 12px; justify-content: center; flex-wrap: wrap; }
        .quiz-buttons.show { display: flex; }
        .quiz-btn {
            padding: 14px 28px;
            font-size: 1em;
            font-weight: 600;
            min-width: 140px;
        }
        .quiz-btn:nth-child(1) { background: #e57373; border-color: #e57373; color: #000; }
        .quiz-btn:nth-child(2) { background: #4fc3f7; border-color: #4fc3f7; color: #000; }
        .quiz-btn:nth-child(3) { background: #81c784; border-color: #81c784; color: #000; }
        .quiz-btn:nth-child(4) { background: #ffb74d; border-color: #ffb74d; color: #000; }

        /* Feedback */
        .feedback {
            text-align: center;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            display: none;
        }
        .feedback.correct { display: block; background: rgba(129,199,132,0.2); color: var(--success); }
        .feedback.incorrect { display: block; background: rgba(229,115,115,0.2); color: var(--error); }

        /* Progress bar */
        .progress-bar {
            height: 4px;
            background: var(--border);
            margin-bottom: 12px;
            border-radius: 2px;
            overflow: hidden;
        }
        .progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }

        /* Modal */
        .modal {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.9);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 200;
        }
        .modal.show { display: flex; }
        .modal-content {
            background: var(--bg-secondary);
            padding: 40px;
            border-radius: 16px;
            text-align: center;
            max-width: 400px;
        }
        .modal h2 { margin-bottom: 16px; color: var(--accent); }
        .modal .score-big { font-size: 3em; font-weight: 700; color: var(--success); margin: 24px 0; }

        /* Stats panel */
        .stats-panel {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            font-size: 0.85em;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid var(--border);
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-row span:first-child { color: var(--text-secondary); }
        .stat-row span:last-child { color: var(--accent); font-weight: 600; }

        /* Comparison progress */
        .comparison-progress {
            position: absolute;
            bottom: 8px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
        }

        /* Loading overlay */
        .loading-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.95);
            display: none;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 300;
        }
        .loading-overlay.show { display: flex; }
        .loading-content {
            text-align: center;
            max-width: 400px;
        }
        .loading-title {
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 16px;
        }
        .loading-status {
            color: var(--text-secondary);
            margin-bottom: 24px;
        }
        .loading-bar {
            width: 300px;
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 16px;
        }
        .loading-bar-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.3s;
            border-radius: 4px;
        }
        .loading-percent {
            font-size: 2em;
            font-weight: 700;
            color: var(--accent);
        }

        /* Action buttons */
        .interval-btn, .clip-btn, .regen-btn {
            font-weight: 600;
            padding: 8px 14px;
            margin-right: 8px;
            font-size: 0.85em;
            border-radius: 6px;
        }
        .interval-btn {
            background: var(--success);
            border-color: var(--success);
            color: #000;
        }
        .clip-btn {
            background: var(--accent);
            border-color: var(--accent);
            color: #000;
        }
        .regen-btn {
            background: var(--warning);
            border-color: var(--warning);
            color: #000;
        }
        .interval-btn:hover, .clip-btn:hover, .regen-btn:hover { opacity: 0.9; }
        .interval-btn:disabled, .clip-btn:disabled, .regen-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Enhanced details overlay */
        .detail-card {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
        }
        .detail-card .label {
            font-size: 0.75em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .detail-card .value {
            font-size: 1.2em;
            font-weight: 600;
            color: var(--accent);
        }
        .detail-card .quality-tier {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 600;
            margin-left: 8px;
        }
        .tier-excellent { background: var(--success); color: #000; }
        .tier-good { background: #81c784; color: #000; }
        .tier-fair { background: var(--warning); color: #000; }
        .tier-poor { background: var(--error); color: #000; }
        .detail-card .explain {
            font-size: 0.75em;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        /* Video quality optimizations */
        video {
            image-rendering: high-quality;
            -webkit-transform: translateZ(0);
        }

        /* Results View */
        .results-view {
            flex: 1;
            padding: 24px;
            overflow-y: auto;
            background: var(--bg);
        }
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .results-header h2 {
            margin: 0;
            color: var(--text);
        }
        .refresh-btn {
            padding: 8px 16px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .results-summary {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .summary-card {
            background: var(--card);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .summary-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--accent);
        }
        .summary-label {
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-top: 4px;
        }
        .results-table-container {
            background: var(--card);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 24px;
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
        }
        .results-table th, .results-table td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        .results-table th {
            background: var(--sidebar);
            color: var(--text);
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        .results-table tr:hover {
            background: var(--hover);
        }
        .results-table .quality-high { color: #22c55e; }
        .results-table .quality-medium { color: #eab308; }
        .results-table .quality-low { color: #f97316; }
        .results-chart {
            background: var(--card);
            padding: 20px;
            border-radius: 12px;
        }
        .results-chart h3 {
            margin: 0 0 16px 0;
            color: var(--text);
        }
        .chart-area {
            min-height: 200px;
        }
        .chart-container {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        .chart-group {
            background: var(--bg);
            padding: 16px;
            border-radius: 8px;
        }
        .chart-group h4 {
            margin: 0 0 12px 0;
            color: var(--accent);
            font-size: 0.9em;
        }
        .chart-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            gap: 12px;
        }
        .chart-label {
            width: 140px;
            font-size: 0.85em;
            color: var(--text-secondary);
            flex-shrink: 0;
        }
        .chart-bar {
            height: 24px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            padding: 0 8px;
            font-size: 0.8em;
            color: #fff;
            font-weight: 600;
            min-width: 50px;
        }
        .chart-bar.bar-good { background: linear-gradient(90deg, #22c55e, #16a34a); }
        .chart-bar.bar-ok { background: linear-gradient(90deg, #eab308, #ca8a04); }
        .chart-bar.bar-low { background: linear-gradient(90deg, #f97316, #ea580c); }
        .chart-time {
            width: 50px;
            font-size: 0.8em;
            color: var(--text-secondary);
            text-align: right;
        }
        .results-table td.good { color: #22c55e; }
        .results-table td.ok { color: #eab308; }
        .results-table td.low { color: #f97316; }

        /* Fullscreen Quiz Mode */
        .quiz-fullscreen {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: #000;
            z-index: 200;
            display: none;
            flex-direction: column;
        }
        .quiz-fullscreen video {
            flex: 1;
            width: 100%;
            object-fit: contain;
            background: #000;
            transition: transform 0.5s ease;  /* Smooth zoom transition */
            transform-origin: center center;
        }
        .quiz-controls {
            background: linear-gradient(transparent, rgba(0,0,0,0.95));
            padding: 24px;
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
        }
        .quiz-progress-bar {
            text-align: center;
            margin-bottom: 16px;
            font-size: 1.1em;
            color: var(--text-secondary);
        }
        .quiz-progress-bar strong {
            color: var(--accent);
            font-size: 1.3em;
        }
        .quiz-feedback {
            text-align: center;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1.2em;
            margin-bottom: 16px;
            min-height: 48px;
        }
        .quiz-feedback.correct {
            background: rgba(129,199,132,0.3);
            color: var(--success);
        }
        .quiz-feedback.incorrect {
            background: rgba(229,115,115,0.3);
            color: var(--error);
        }
        .quiz-buttons-container {
            display: flex;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
        }
        .quiz-buttons-container .quiz-btn {
            padding: 16px 32px;
            font-size: 1.1em;
            font-weight: 600;
            min-width: 160px;
            border-radius: 8px;
            transition: transform 0.2s, opacity 0.2s;
        }
        .quiz-buttons-container .quiz-btn:hover {
            transform: scale(1.05);
        }
        .quiz-buttons-container .quiz-btn:active {
            transform: scale(0.98);
        }
        .quiz-exit-btn {
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(255,255,255,0.1);
            border: none;
            color: #fff;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .quiz-exit-btn:hover {
            background: rgba(255,255,255,0.2);
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <button class="icon-btn" onclick="toggleSidebar()" title="Toggle Sidebar">☰</button>
            <h1>VFI+SR Blind Study</h1>
        </div>
        <div class="header-stats">
            <span class="stat">Resolution: <strong id="resolution">-</strong></span>
            <span class="stat">FPS: <strong id="fps">-</strong></span>
            <span class="stat">Clip: <strong id="clipNum">-</strong></span>
            <span class="stat" id="scoreDisplay"></span>
        </div>
        <div class="header-controls">
            <button class="interval-btn" onclick="newInterval()" title="New random 10s interval (fast if pre-generated)">🎲 New Interval</button>
            <button class="clip-btn" onclick="newClip()" title="Switch to different source video">📁 New Clip</button>
            <button class="regen-btn" onclick="regenerateAll()" title="Full regeneration from scratch (slow)">🔄 Regenerate</button>
            <button class="icon-btn" onclick="toggleTheme()" title="Toggle Theme" id="themeBtn">🌙</button>
        </div>
    </div>

    <div class="main">
        <div class="sidebar" id="sidebar">
            <div class="phase-toggle">
                <button class="phase-btn active" id="learnBtn" onclick="setPhase('learn')">Learn</button>
                <button class="phase-btn" id="comparisonBtn" onclick="setPhase('comparison')">Compare</button>
                <button class="phase-btn" id="quizBtn" onclick="setPhase('quiz')">Quiz</button>
                <button class="phase-btn" id="resultsBtn" onclick="setPhase('results')">Results</button>
            </div>

            <h3>Models</h3>
            <div class="model-list" id="modelList"></div>

            <h3>Performance Stats</h3>
            <div id="statsPanel" class="stats-panel">
                <div class="stat-row"><span>Processing:</span><span id="statTime">-</span></div>
                <div class="stat-row"><span>PSNR:</span><span id="statPsnr">-</span></div>
                <div class="stat-row"><span>SSIM:</span><span id="statSsim">-</span></div>
                <div class="stat-row"><span>VFI Method:</span><span id="statVfi">-</span></div>
            </div>
        </div>

        <div class="video-area">
            <!-- Single video view (Learn/Quiz) -->
            <div class="single-view" id="singleView">
                <video id="video" controls loop preload="auto" playsinline>
                    <source id="videoSource" src="" type="video/mp4">
                </video>
                <div class="video-label" id="videoLabel">-</div>
                <div class="details-overlay" id="detailsOverlay">
                    <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                        <div class="detail-card">
                            <div class="label">Quality vs Original</div>
                            <div class="value"><span id="detailPsnr">-</span> <span id="psnrTier" class="quality-tier"></span></div>
                            <div class="explain">Higher dB = closer to original</div>
                        </div>
                        <div class="detail-card">
                            <div class="label">Visual Similarity</div>
                            <div class="value"><span id="detailSsim">-</span> <span id="ssimTier" class="quality-tier"></span></div>
                            <div class="explain">How structurally similar to reference</div>
                        </div>
                        <div class="detail-card">
                            <div class="label">Bitrate</div>
                            <div class="value"><span id="detailBitrate">-</span></div>
                            <div class="explain">Data rate - higher = larger files, better quality</div>
                        </div>
                        <div class="detail-card">
                            <div class="label">File Size</div>
                            <div class="value"><span id="detailFileSize">-</span></div>
                            <div class="explain">Total video file size</div>
                        </div>
                        <div class="detail-card">
                            <div class="label">Processing Time</div>
                            <div class="value"><span id="detailTime">-</span></div>
                            <div class="explain">Time to generate this clip</div>
                        </div>
                        <div class="detail-card">
                            <div class="label">VFI Method</div>
                            <div class="value"><span id="detailVfi">-</span></div>
                            <div class="explain" id="detailVfiExplain">Frame interpolation technique</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Comparison view (side-by-side) -->
            <div class="comparison-view" id="comparisonView">
                <div class="video-panel">
                    <video id="controlVideo" muted loop></video>
                    <div class="panel-label control">CONTROL (Reference)</div>
                </div>
                <div class="video-panel">
                    <video id="methodVideo" muted loop></video>
                    <div class="panel-label method" id="methodLabel">-</div>
                </div>
                <div class="comparison-progress" id="comparisonProgress">1 / 3</div>
            </div>

            <!-- Results view (experiment data visualization) -->
            <div class="results-view" id="resultsView" style="display:none">
                <div class="results-header">
                    <h2>Experiment Results</h2>
                    <button class="refresh-btn" onclick="loadExperimentResults()">↻ Refresh</button>
                </div>
                <div class="results-summary" id="resultsSummary">
                    <div class="summary-card">
                        <div class="summary-value" id="totalExperiments">0</div>
                        <div class="summary-label">Total Experiments</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value" id="totalMethods">0</div>
                        <div class="summary-label">Methods Tested</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-value" id="totalQualities">0</div>
                        <div class="summary-label">Quality Levels</div>
                    </div>
                </div>
                <div class="results-table-container">
                    <table class="results-table" id="resultsTable">
                        <thead>
                            <tr>
                                <th>Method</th>
                                <th>Quality</th>
                                <th>PSNR (dB)</th>
                                <th>SSIM</th>
                                <th>Time (s)</th>
                                <th>Resolution</th>
                                <th>FPS</th>
                            </tr>
                        </thead>
                        <tbody id="resultsTableBody"></tbody>
                    </table>
                </div>
                <div class="results-chart" id="resultsChart">
                    <h3>Quality vs Processing Time</h3>
                    <div class="chart-area" id="chartArea"></div>
                </div>
            </div>

            <div class="controls-bar">
                <div class="feedback" id="feedback"></div>
                <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>

                <div class="controls-row" id="normalControls">
                    <button onclick="prevClip()">← Prev</button>
                    <button class="primary" onclick="playVideo()">▶ Play</button>
                    <button onclick="nextClip()">Next →</button>
                    <button onclick="toggleDetails()">📊 Details</button>
                </div>

                <div class="controls-row" id="comparisonControls" style="display:none">
                    <button onclick="restartComparison()">↺ Restart</button>
                    <button class="primary" onclick="playComparison()">▶ Play Comparison</button>
                    <button onclick="skipToNext()">Skip →</button>
                </div>

                <div class="quiz-buttons" id="quizButtons"></div>
            </div>
        </div>
    </div>

    <div class="modal" id="resultsModal">
        <div class="modal-content">
            <h2>Quiz Complete!</h2>
            <div class="score-big" id="finalScore">0/0</div>
            <p id="resultMsg"></p>
            <button class="primary" onclick="closeResults()" style="margin-top:24px">Continue</button>
        </div>
    </div>

    <!-- Loading overlay for re-processing -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-content">
            <div class="loading-title">Generating New Clips</div>
            <div class="loading-status" id="loadingStatus">Starting...</div>
            <div class="loading-bar">
                <div class="loading-bar-fill" id="loadingBarFill" style="width: 0%"></div>
            </div>
            <div class="loading-percent" id="loadingPercent">0%</div>
            <p style="margin-top: 24px; color: var(--text-secondary); font-size: 0.85em;">
                Processing 5 methods: control, degraded, lanczos, rife_lanczos, adaptive_vfi
            </p>
        </div>
    </div>

    <!-- Fullscreen Quiz Mode -->
    <div class="quiz-fullscreen" id="quizFullscreen">
        <button class="quiz-exit-btn" onclick="exitQuiz()">Exit Quiz</button>
        <video id="quizVideo" controls loop preload="auto" playsinline></video>
        <div class="quiz-controls">
            <div class="quiz-progress-bar">
                Question <strong id="quizQuestionNum">1</strong> of <strong id="quizTotalQuestions">5</strong>
                &nbsp;|&nbsp; Score: <strong id="quizScoreLive">0</strong>
            </div>
            <div class="quiz-feedback" id="quizFeedback"></div>
            <div class="quiz-buttons-container" id="quizButtonsFullscreen"></div>
        </div>
    </div>

    <script>
        let state = {
            phase: 'learn',
            currentIdx: 0,
            clips: [],
            models: [],
            quizOrder: [],
            quizScore: 0,
            quizTotal: 0,
            correctlyGuessed: [],  // Track models correctly guessed in quiz
            metadata: null,
            comparisonIdx: 0,
            comparisonMethods: [],
            sidebarCollapsed: false,
            sidebarWasCollapsed: false,  // Track sidebar state before quiz
            darkMode: true,
            showDetails: false
        };

        document.addEventListener('DOMContentLoaded', () => {
            loadData();
            loadPreferences();
        });

        function loadPreferences() {
            const dark = localStorage.getItem('darkMode');
            if (dark === 'false') toggleTheme();
            const sidebar = localStorage.getItem('sidebarCollapsed');
            if (sidebar === 'true') toggleSidebar();
        }

        async function loadData() {
            const res = await fetch('/api/metadata');
            state.metadata = await res.json();
            state.clips = state.metadata.clips || [];
            state.models = state.metadata.models || [];
            state.comparisonMethods = state.clips.filter(c => !c.is_control);

            document.getElementById('resolution').textContent = state.metadata.resolution || '-';
            document.getElementById('fps').textContent = (state.metadata.fps || '-') + 'fps';

            buildModelList();
            buildQuizButtons();
            updateDisplay();

            // Preload control video for comparison
            const control = state.clips.find(c => c.is_control);
            if (control) {
                document.getElementById('controlVideo').src = '/api/video/' + control.model;
            }
        }

        function toggleTheme() {
            state.darkMode = !state.darkMode;
            document.body.classList.toggle('light-mode', !state.darkMode);
            document.getElementById('themeBtn').textContent = state.darkMode ? '🌙' : '☀️';
            localStorage.setItem('darkMode', state.darkMode);
        }

        function toggleSidebar() {
            state.sidebarCollapsed = !state.sidebarCollapsed;
            document.getElementById('sidebar').classList.toggle('collapsed', state.sidebarCollapsed);
            localStorage.setItem('sidebarCollapsed', state.sidebarCollapsed);
        }

        function toggleDetails() {
            state.showDetails = !state.showDetails;
            document.getElementById('detailsOverlay').classList.toggle('show', state.showDetails);
        }

        function buildModelList() {
            const list = document.getElementById('modelList');
            list.innerHTML = '';
            state.clips.forEach((clip, i) => {
                const div = document.createElement('div');
                const isInnovative = clip.is_novel || ['uafi_default', 'ughi_default', 'mcar_default', 'mcar_aggressive'].includes(clip.model);
                const isDegraded = clip.is_degraded || clip.model === 'degraded';
                div.className = 'model-item' + (clip.is_control ? ' control' : '') + (isInnovative ? ' innovative' : '');
                div.dataset.index = i;
                const suffix = clip.is_control ? ' (Ref)' : isDegraded ? ' (Baseline)' : isInnovative ? ' ★' : '';
                div.innerHTML = `
                    <div class="name">${formatModel(clip.model)}${suffix}</div>
                    <div class="details">${clip.vfi_method || '-'} | ${clip.bitrate_mbps ? clip.bitrate_mbps.toFixed(1) + ' Mbps' : clip.psnr_db ? clip.psnr_db + 'dB' : '-'}</div>
                `;
                div.onclick = () => { state.currentIdx = i; updateDisplay(); };
                list.appendChild(div);
            });
        }

        function buildQuizButtons(targetContainer = 'quizButtons') {
            const container = document.getElementById(targetContainer);
            container.innerHTML = '';
            // Include ALL models EXCEPT ones already correctly guessed
            const availableModels = state.models.filter(m => !state.correctlyGuessed.includes(m));
            availableModels.forEach((model) => {
                const btn = document.createElement('button');
                btn.className = 'quiz-btn';
                btn.textContent = formatModel(model);
                btn.onclick = () => submitGuess(model);
                container.appendChild(btn);
            });
        }

        function formatModel(name) {
            if (!name) return '-';
            return name.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
        }

        function setPhase(phase) {
            const prevPhase = state.phase;
            state.phase = phase;
            state.currentIdx = 0;
            state.quizScore = 0;
            state.quizTotal = 0;
            state.comparisonIdx = 0;

            document.querySelectorAll('.phase-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(phase + 'Btn')?.classList.add('active');

            const singleView = document.getElementById('singleView');
            const comparisonView = document.getElementById('comparisonView');
            const resultsView = document.getElementById('resultsView');
            const normalControls = document.getElementById('normalControls');
            const comparisonControls = document.getElementById('comparisonControls');
            const quizBtns = document.getElementById('quizButtons');
            const sidebar = document.getElementById('sidebar');
            const controlsBar = document.querySelector('.controls-bar');

            // Hide all views first
            singleView.style.display = 'none';
            comparisonView.classList.remove('active');
            resultsView.style.display = 'none';
            normalControls.style.display = 'none';
            comparisonControls.style.display = 'none';
            quizBtns.classList.remove('show');
            if (controlsBar) controlsBar.style.display = 'flex';

            if (phase === 'results') {
                resultsView.style.display = 'block';
                if (controlsBar) controlsBar.style.display = 'none';
                loadExperimentResults();
                return;
            } else if (phase === 'comparison') {
                comparisonView.classList.add('active');
                comparisonControls.style.display = 'flex';
                state.comparisonMethods = state.clips.filter(c => !c.is_control);
                updateComparison();
            } else {
                singleView.style.display = 'flex';
                normalControls.style.display = 'flex';

                if (phase === 'quiz') {
                    // Reset correctly guessed for new quiz
                    state.correctlyGuessed = [];
                    // Show fullscreen quiz mode
                    document.getElementById('quizFullscreen').style.display = 'flex';
                    document.querySelector('.header').style.display = 'none';
                    singleView.style.display = 'none';
                    quizBtns.classList.remove('show');  // Hide old buttons
                    // Include ALL clips in quiz (including control)
                    state.quizOrder = [...state.clips].sort(() => Math.random() - 0.5);
                    buildQuizButtons('quizButtonsFullscreen');
                    updateQuizDisplay();
                    return;  // Don't call updateDisplay for quiz mode
                } else {
                    // Restore sidebar state when leaving quiz
                    if (prevPhase === 'quiz' && !state.sidebarWasCollapsed) {
                        sidebar.classList.remove('collapsed');
                        state.sidebarCollapsed = false;
                    }
                    quizBtns.classList.remove('show');
                    state.quizOrder = [...state.clips];
                }
                updateDisplay();
            }
            updateScore();
        }

        function updateComparison() {
            const methods = state.comparisonMethods;
            if (state.comparisonIdx >= methods.length) {
                state.comparisonIdx = 0;
            }
            const method = methods[state.comparisonIdx];
            document.getElementById('methodVideo').src = '/api/video/' + method.model;
            document.getElementById('methodLabel').textContent = formatModel(method.model);
            document.getElementById('comparisonProgress').textContent =
                `${state.comparisonIdx + 1} / ${methods.length}`;

            // Update stats
            updateStats(method);
        }

        function playComparison() {
            const controlVid = document.getElementById('controlVideo');
            const methodVid = document.getElementById('methodVideo');

            controlVid.currentTime = 0;
            methodVid.currentTime = 0;
            controlVid.playbackRate = 0.25;
            methodVid.playbackRate = 0.25;

            controlVid.play();
            methodVid.play();

            // Auto-advance when video ends (5s at 0.25x = 20s real time, but video is 10s so 2.5s content)
            methodVid.onended = () => {
                state.comparisonIdx++;
                if (state.comparisonIdx < state.comparisonMethods.length) {
                    updateComparison();
                    setTimeout(playComparison, 500);
                }
            };
        }

        function restartComparison() {
            state.comparisonIdx = 0;
            updateComparison();
        }

        function skipToNext() {
            const controlVid = document.getElementById('controlVideo');
            const methodVid = document.getElementById('methodVideo');
            controlVid.pause();
            methodVid.pause();
            state.comparisonIdx++;
            if (state.comparisonIdx >= state.comparisonMethods.length) {
                state.comparisonIdx = 0;
            }
            updateComparison();
        }

        function getClipList() {
            if (state.phase === 'quiz') return state.quizOrder;
            return state.clips;
        }

        function updateDisplay() {
            const clips = getClipList();
            if (!clips.length) return;

            const clip = clips[state.currentIdx];
            const video = document.getElementById('video');
            document.getElementById('videoSource').src = '/api/video/' + clip.model;
            video.load();

            // Update label
            const label = document.getElementById('videoLabel');
            if (state.phase === 'quiz') {
                label.textContent = '???';
                label.className = 'video-label';
            } else {
                label.textContent = formatModel(clip.model);
                label.className = 'video-label ' + (clip.is_control ? 'control' : 'method');
            }

            // Highlight in sidebar
            document.querySelectorAll('.model-item').forEach((item, i) => {
                item.classList.toggle('active', state.clips[i]?.model === clip.model);
            });

            // Progress
            document.getElementById('clipNum').textContent = `${state.currentIdx + 1}/${clips.length}`;
            document.getElementById('progressFill').style.width = ((state.currentIdx + 1) / clips.length * 100) + '%';

            // Stats
            updateStats(clip);

            // Enhanced details overlay
            updateDetailsOverlay(clip);

            document.getElementById('feedback').className = 'feedback';
        }

        function updateDetailsOverlay(clip) {
            // PSNR with quality tier
            const psnr = clip.psnr_db || 0;
            document.getElementById('detailPsnr').textContent = psnr ? psnr.toFixed(2) + ' dB' : '-';
            const psnrTier = document.getElementById('psnrTier');
            if (psnr >= 40) {
                psnrTier.textContent = 'Excellent';
                psnrTier.className = 'quality-tier tier-excellent';
            } else if (psnr >= 30) {
                psnrTier.textContent = 'Good';
                psnrTier.className = 'quality-tier tier-good';
            } else if (psnr >= 20) {
                psnrTier.textContent = 'Fair';
                psnrTier.className = 'quality-tier tier-fair';
            } else {
                psnrTier.textContent = 'Poor';
                psnrTier.className = 'quality-tier tier-poor';
            }

            // SSIM with quality tier (as percentage)
            const ssim = clip.ssim || 0;
            document.getElementById('detailSsim').textContent = ssim ? (ssim * 100).toFixed(1) + '%' : '-';
            const ssimTier = document.getElementById('ssimTier');
            if (ssim >= 0.95) {
                ssimTier.textContent = 'Excellent';
                ssimTier.className = 'quality-tier tier-excellent';
            } else if (ssim >= 0.85) {
                ssimTier.textContent = 'Good';
                ssimTier.className = 'quality-tier tier-good';
            } else if (ssim >= 0.70) {
                ssimTier.textContent = 'Fair';
                ssimTier.className = 'quality-tier tier-fair';
            } else {
                ssimTier.textContent = 'Poor';
                ssimTier.className = 'quality-tier tier-poor';
            }

            // Bitrate
            document.getElementById('detailBitrate').textContent = clip.bitrate_mbps ? clip.bitrate_mbps.toFixed(1) + ' Mbps' : '-';

            // File size
            document.getElementById('detailFileSize').textContent = clip.file_size_mb ? clip.file_size_mb.toFixed(1) + ' MB' : '-';

            // Processing time
            document.getElementById('detailTime').textContent = clip.processing_time ? clip.processing_time.toFixed(1) + 's' : '-';

            // VFI method with explanation
            const vfiMethod = clip.vfi_method || '-';
            document.getElementById('detailVfi').textContent = formatModel(vfiMethod);
            const vfiExplain = document.getElementById('detailVfiExplain');
            switch(vfiMethod) {
                case 'none':
                    vfiExplain.textContent = 'No interpolation - original reference';
                    break;
                case 'frame_dup':
                    vfiExplain.textContent = 'Frame duplication - shows judder artifacts';
                    break;
                case 'linear_blend':
                    vfiExplain.textContent = 'Simple blending - fast but ghosting';
                    break;
                case 'RIFE':
                    vfiExplain.textContent = 'Neural network VFI - high quality, slow';
                    break;
                case 'adaptive':
                    vfiExplain.textContent = 'Motion-aware hybrid - smart + efficient';
                    break;
                case 'UAFI':
                    vfiExplain.textContent = 'UI-Aware - Detects HUD regions, preserves without ghosting';
                    break;
                case 'UGHI':
                    vfiExplain.textContent = 'Uncertainty-Guided - Bidirectional flow for selective refinement';
                    break;
                case 'MCAR':
                    vfiExplain.textContent = 'Motion-Complexity Adaptive - Routes frames by difficulty';
                    break;
                default:
                    vfiExplain.textContent = 'Frame interpolation technique';
            }
        }

        function updateStats(clip) {
            document.getElementById('statTime').textContent = clip.processing_time ? clip.processing_time.toFixed(1) + 's' : '-';
            document.getElementById('statPsnr').textContent = clip.psnr_db ? clip.psnr_db.toFixed(2) + ' dB' : '-';
            document.getElementById('statSsim').textContent = clip.ssim ? clip.ssim.toFixed(4) : '-';
            document.getElementById('statVfi').textContent = clip.vfi_method || '-';
        }

        function playVideo() {
            document.getElementById('video').play();
        }

        function prevClip() {
            const clips = getClipList();
            if (state.currentIdx > 0) {
                state.currentIdx--;
                updateDisplay();
            }
        }

        function nextClip() {
            const clips = getClipList();
            if (state.currentIdx < clips.length - 1) {
                state.currentIdx++;
                updateDisplay();
            }
        }

        function submitGuess(guess) {
            if (state.phase !== 'quiz') return;

            const clip = state.quizOrder[state.currentIdx];
            const correct = guess === clip.model;

            state.quizTotal++;
            if (correct) {
                state.quizScore++;
                // Add to correctly guessed list and rebuild buttons
                state.correctlyGuessed.push(clip.model);
                buildQuizButtons('quizButtonsFullscreen');
            }
            updateScore();

            // Update fullscreen feedback
            const feedback = document.getElementById('quizFeedback');
            feedback.textContent = correct
                ? '✓ Correct! ' + formatModel(clip.model)
                : '✗ Wrong! It was ' + formatModel(clip.model);
            feedback.className = 'quiz-feedback ' + (correct ? 'correct' : 'incorrect');

            // Update live score
            document.getElementById('quizScoreLive').textContent = state.quizScore;

            // Disable buttons during feedback
            const buttons = document.querySelectorAll('#quizButtonsFullscreen .quiz-btn');
            buttons.forEach(btn => btn.disabled = true);

            setTimeout(() => {
                feedback.textContent = '';
                feedback.className = 'quiz-feedback';
                buttons.forEach(btn => btn.disabled = false);

                if (state.currentIdx < state.quizOrder.length - 1) {
                    state.currentIdx++;
                    updateQuizDisplay();
                } else {
                    showResults();
                }
            }, 1500);
        }

        function updateScore() {
            const el = document.getElementById('scoreDisplay');
            if (state.phase === 'quiz' && state.quizTotal > 0) {
                const pct = Math.round(state.quizScore / state.quizTotal * 100);
                el.innerHTML = `Score: <strong>${state.quizScore}/${state.quizTotal}</strong> (${pct}%)`;
            } else {
                el.textContent = '';
            }
        }

        function updateQuizDisplay() {
            const clips = state.quizOrder;
            if (!clips.length) return;

            const clip = clips[state.currentIdx];
            const video = document.getElementById('quizVideo');

            // Remove old listener before setting new source
            video.removeEventListener('timeupdate', handleQuizPlayback);

            video.src = '/api/video/' + clip.model;
            video.load();

            // Reset playback state for new video
            video.playbackRate = 1.0;
            video.style.transform = 'scale(1)';

            // Add playback phase handler (normal → zoom+slow-mo → normal)
            video.addEventListener('timeupdate', handleQuizPlayback);

            video.play();

            // Update progress
            document.getElementById('quizQuestionNum').textContent = state.currentIdx + 1;
            document.getElementById('quizTotalQuestions').textContent = clips.length;
            document.getElementById('quizScoreLive').textContent = state.quizScore;
        }

        // Quiz video playback phases: normal (1/3) → zoom+slow-mo (1/3) → normal (1/3)
        function handleQuizPlayback() {
            const video = document.getElementById('quizVideo');
            if (!video.duration) return;

            const progress = video.currentTime / video.duration;

            if (progress < 0.33) {
                // First third: normal playback
                if (video.playbackRate !== 1.0) {
                    video.playbackRate = 1.0;
                    video.style.transform = 'scale(1)';
                }
            } else if (progress < 0.66) {
                // Middle third: zoom in + slow-mo
                if (video.playbackRate !== 0.25) {
                    video.playbackRate = 0.25;
                    video.style.transform = 'scale(2)';
                }
            } else {
                // Last third: normal playback
                if (video.playbackRate !== 1.0) {
                    video.playbackRate = 1.0;
                    video.style.transform = 'scale(1)';
                }
            }
        }

        function exitQuiz() {
            // Hide fullscreen quiz
            document.getElementById('quizFullscreen').style.display = 'none';
            // Show header
            document.querySelector('.header').style.display = 'flex';
            // Pause quiz video
            document.getElementById('quizVideo').pause();
            // Switch to learn mode
            setPhase('learn');
        }

        function showResults() {
            const pct = Math.round(state.quizScore / state.quizTotal * 100);
            document.getElementById('finalScore').textContent = `${state.quizScore}/${state.quizTotal}`;
            let msg = pct === 100 ? 'Perfect! You can distinguish all models.'
                : pct >= 75 ? 'Excellent eye for quality differences!'
                : pct >= 50 ? 'Good! Some models are quite similar.'
                : 'The differences are subtle at this resolution.';
            document.getElementById('resultMsg').textContent = msg;
            // Hide fullscreen quiz and show modal
            document.getElementById('quizFullscreen').style.display = 'none';
            document.getElementById('quizVideo').pause();
            document.getElementById('resultsModal').classList.add('show');
        }

        function closeResults() {
            document.getElementById('resultsModal').classList.remove('show');
            // Show header again
            document.querySelector('.header').style.display = 'flex';
            // Restore sidebar when quiz ends
            if (!state.sidebarWasCollapsed) {
                document.getElementById('sidebar').classList.remove('collapsed');
                state.sidebarCollapsed = false;
            }
            setPhase('learn');
        }

        // 3-button system: New Interval (fast), New Clip, Regenerate (slow)
        async function newInterval() {
            // Fast clip from pre-generated video (if available)
            setButtonsEnabled(false);
            showLoadingOverlay('Extracting new interval...');
            try {
                const res = await fetch('/api/new-interval', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'done') {
                    // Fast extraction complete
                    await loadData();
                    setPhase('learn');
                    hideLoadingOverlay();
                } else if (data.status === 'fallback') {
                    // No pre-generated videos, fall back to regeneration
                    document.getElementById('loadingStatus').textContent = 'No pre-generated videos. Running full processing...';
                    pollProgress();
                } else {
                    alert('Error: ' + (data.error || 'Failed to extract interval'));
                    hideLoadingOverlay();
                }
            } catch (err) {
                alert('Error: ' + err.message);
                hideLoadingOverlay();
            }
            setButtonsEnabled(true);
        }

        async function newClip() {
            // Switch to a different source video
            setButtonsEnabled(false);
            try {
                const res = await fetch('/api/available-clips');
                const data = await res.json();

                if (data.clips && data.clips.length > 1) {
                    const current = data.current || '';
                    const others = data.clips.filter(c => c !== current);
                    if (others.length > 0) {
                        showLoadingOverlay('Loading new clip...');
                        const selected = others[Math.floor(Math.random() * others.length)];
                        const switchRes = await fetch('/api/switch-clip', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ clip: selected })
                        });
                        const switchData = await switchRes.json();
                        if (switchData.status === 'done') {
                            await loadData();
                            setPhase('learn');
                        } else {
                            alert('Error: ' + (switchData.error || 'Failed to switch clip'));
                        }
                        hideLoadingOverlay();
                    } else {
                        alert('Only one source clip available.');
                    }
                } else {
                    alert('Only one source clip available. Add more videos to data/raw/ to enable this feature.');
                }
            } catch (err) {
                alert('Error: ' + err.message);
            }
            setButtonsEnabled(true);
        }

        async function regenerateAll() {
            if (!confirm('Regenerate all clips from scratch?\\n\\nThis will take several minutes to process.')) {
                return;
            }
            setButtonsEnabled(false);
            showLoadingOverlay('Regenerating all clips...');
            try {
                const res = await fetch('/api/reprocess', { method: 'POST' });
                if (!res.ok) {
                    const data = await res.json();
                    alert('Error: ' + (data.error || 'Failed to start processing'));
                    hideLoadingOverlay();
                    setButtonsEnabled(true);
                    return;
                }
                pollProgress();
            } catch (err) {
                alert('Error: ' + err.message);
                hideLoadingOverlay();
                setButtonsEnabled(true);
            }
        }

        function setButtonsEnabled(enabled) {
            document.querySelectorAll('.interval-btn, .clip-btn, .regen-btn').forEach(btn => {
                btn.disabled = !enabled;
            });
        }

        function showLoadingOverlay(message = 'Starting...') {
            document.getElementById('loadingOverlay').classList.add('show');
            document.getElementById('loadingBarFill').style.width = '0%';
            document.getElementById('loadingPercent').textContent = '0%';
            document.getElementById('loadingStatus').textContent = message;
        }

        function hideLoadingOverlay() {
            document.getElementById('loadingOverlay').classList.remove('show');
        }

        async function pollProgress() {
            try {
                const res = await fetch('/api/processing-status');
                const data = await res.json();

                document.getElementById('loadingBarFill').style.width = data.progress + '%';
                document.getElementById('loadingPercent').textContent = data.progress + '%';
                document.getElementById('loadingStatus').textContent = data.status || 'Processing...';

                if (data.error) {
                    alert('Processing error: ' + data.error);
                    hideLoadingOverlay();
                    return;
                }

                if (data.progress >= 100 || !data.is_processing) {
                    if (data.progress >= 100) {
                        // Reload data
                        await loadData();
                        setPhase('learn');
                    }
                    hideLoadingOverlay();
                    return;
                }

                // Continue polling
                setTimeout(pollProgress, 1000);
            } catch (err) {
                console.error('Poll error:', err);
                setTimeout(pollProgress, 2000);
            }
        }

        async function loadExperimentResults() {
            try {
                const res = await fetch('/api/experiment-results');
                const data = await res.json();

                if (data.error) {
                    document.getElementById('resultsTableBody').innerHTML =
                        '<tr><td colspan="7" style="text-align:center;color:#f66;">No experiment data found</td></tr>';
                    return;
                }

                const results = data.results || [];

                // Update summary cards
                document.getElementById('totalExperiments').textContent = results.length;
                const methods = new Set(results.map(r => r.method_base || r.method));
                document.getElementById('totalMethods').textContent = methods.size;
                const qualities = new Set(results.map(r => r.quality_level || 'default'));
                document.getElementById('totalQualities').textContent = qualities.size;

                // Build table rows
                const tbody = document.getElementById('resultsTableBody');
                tbody.innerHTML = '';

                // Sort by quality level then method
                results.sort((a, b) => {
                    const qOrder = {'high': 0, 'medium': 1, 'low': 2};
                    const qa = qOrder[a.quality_level] ?? 3;
                    const qb = qOrder[b.quality_level] ?? 3;
                    if (qa !== qb) return qa - qb;
                    return (a.method || '').localeCompare(b.method || '');
                });

                for (const r of results) {
                    const tr = document.createElement('tr');
                    const psnrClass = r.psnr_db > 20 ? 'good' : r.psnr_db > 15 ? 'ok' : 'low';
                    const ssimClass = r.ssim > 0.7 ? 'good' : r.ssim > 0.5 ? 'ok' : 'low';

                    tr.innerHTML = `
                        <td><strong>${formatModel(r.method_base || r.method)}</strong></td>
                        <td>${r.quality_level || '-'}</td>
                        <td class="${psnrClass}">${(r.psnr_db || 0).toFixed(2)}</td>
                        <td class="${ssimClass}">${(r.ssim || 0).toFixed(4)}</td>
                        <td>${(r.processing_time_s || 0).toFixed(1)}</td>
                        <td>${r.target_resolution || '-'}</td>
                        <td>${r.target_fps || '-'}</td>
                    `;
                    tbody.appendChild(tr);
                }

                // Build simple text chart (ASCII bar chart)
                buildChart(results);
            } catch (err) {
                console.error('Failed to load experiment results:', err);
                document.getElementById('resultsTableBody').innerHTML =
                    '<tr><td colspan="7" style="text-align:center;color:#f66;">Failed to load data</td></tr>';
            }
        }

        function buildChart(results) {
            const chartArea = document.getElementById('chartArea');
            if (!results.length) {
                chartArea.innerHTML = '<p>No data to display</p>';
                return;
            }

            // Group by quality level for better visualization
            const byQuality = {};
            for (const r of results) {
                const q = r.quality_level || 'default';
                if (!byQuality[q]) byQuality[q] = [];
                byQuality[q].push(r);
            }

            let html = '<div class="chart-container">';

            // Find max PSNR for scaling
            const maxPsnr = Math.max(...results.map(r => r.psnr_db || 0), 30);

            for (const [quality, items] of Object.entries(byQuality)) {
                html += `<div class="chart-group"><h4>${quality.toUpperCase()}</h4>`;

                for (const r of items) {
                    const psnr = r.psnr_db || 0;
                    const width = Math.max(5, (psnr / maxPsnr) * 100);
                    const method = r.method_base || r.method || 'unknown';
                    const barClass = psnr > 20 ? 'bar-good' : psnr > 15 ? 'bar-ok' : 'bar-low';

                    html += `
                        <div class="chart-row">
                            <span class="chart-label">${formatModel(method)}</span>
                            <div class="chart-bar ${barClass}" style="width:${width}%">
                                ${psnr.toFixed(1)}dB
                            </div>
                            <span class="chart-time">${(r.processing_time_s || 0).toFixed(0)}s</span>
                        </div>
                    `;
                }

                html += '</div>';
            }

            html += '</div>';
            chartArea.innerHTML = html;
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', e => {
            if (e.key === 'ArrowLeft') prevClip();
            else if (e.key === 'ArrowRight') nextClip();
            else if (e.key === ' ') {
                e.preventDefault();
                const v = document.getElementById('video');
                v.paused ? v.play() : v.pause();
            }
            else if (e.key === 'd') toggleDetails();
            else if (e.key === 't') toggleTheme();
        });
    </script>
</body>
</html>
'''


def load_video_metadata(video_dir: Path) -> Dict:
    metadata_path = video_dir / 'clips_metadata.json'
    if not metadata_path.exists():
        return {'clips': [], 'models': [], 'error': 'No metadata found'}
    try:
        with open(metadata_path) as f:
            data = json.load(f)
            if 'clips' not in data:
                data['clips'] = []
            if 'models' not in data:
                data['models'] = [c.get('model', 'unknown') for c in data['clips']]
            return data
    except json.JSONDecodeError as e:
        return {'clips': [], 'models': [], 'error': f'Invalid JSON: {e}'}
    except Exception as e:
        return {'clips': [], 'models': [], 'error': f'Failed to load metadata: {e}'}


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/metadata')
def get_metadata():
    if state['metadata'] is None:
        return jsonify({'error': 'Metadata not loaded', 'clips': [], 'models': []}), 500
    if 'error' in state['metadata']:
        return jsonify(state['metadata']), 500
    return jsonify(state['metadata'])


@app.route('/api/experiment-results')
def get_experiment_results():
    """Load experiment results from the outputs directory."""
    project_root = Path(__file__).parent.parent
    results_path = project_root / 'outputs' / 'experiment_results.json'

    if not results_path.exists():
        return jsonify({'error': 'No experiment results found', 'results': []})

    try:
        with open(results_path) as f:
            data = json.load(f)

        results = data.get('results', [])

        # Process results to add method_base (without quality suffix)
        for r in results:
            method = r.get('method', '')
            # Extract base method name (remove _4K@120, _1440p@90, _1080p@60 suffixes)
            import re
            base_match = re.sub(r'_(4K@\d+|1440p@\d+|1080p@\d+)$', '', method)
            r['method_base'] = base_match

        return jsonify({
            'results': results,
            'meta': {
                'total': len(results),
                'last_run': data.get('last_run'),
                'intervals': data.get('intervals', [])
            }
        })
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON: {e}', 'results': []}), 500
    except Exception as e:
        return jsonify({'error': str(e), 'results': []}), 500


def stream_video(video_path: Path):
    try:
        file_size = video_path.stat().st_size
    except OSError:
        return jsonify({'error': 'Cannot read video file'}), 500

    range_header = request.headers.get('Range', None)

    if range_header:
        byte_start, byte_end = 0, None
        try:
            match = range_header.replace('bytes=', '').split('-')
            if match[0]:
                byte_start = int(match[0])
            if len(match) > 1 and match[1]:
                byte_end = int(match[1])
        except (ValueError, IndexError):
            return jsonify({'error': 'Invalid Range header'}), 400

        if byte_start < 0 or byte_start >= file_size:
            return jsonify({'error': 'Range start out of bounds'}), 416
        if byte_end is None:
            byte_end = min(byte_start + 10 * 1024 * 1024, file_size - 1)
        byte_end = min(byte_end, file_size - 1)
        length = byte_end - byte_start + 1

        def generate():
            with open(video_path, 'rb') as f:
                f.seek(byte_start)
                remaining = length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        response = Response(generate(), status=206, mimetype='video/mp4', direct_passthrough=True)
        response.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = length
        return response
    else:
        def generate():
            with open(video_path, 'rb') as f:
                while True:
                    data = f.read(8192)
                    if not data:
                        break
                    yield data
        response = Response(generate(), status=200, mimetype='video/mp4', direct_passthrough=True)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = file_size
        return response


@app.route('/api/video/<model_name>')
def get_video(model_name: str):
    if not model_name or '..' in model_name or '/' in model_name or '\\' in model_name:
        return jsonify({'error': 'Invalid model name'}), 400

    if model_name not in state['models']:
        return jsonify({'error': f'Unknown model: {model_name}', 'valid_models': state['models']}), 404

    video_dir = state['video_dir']
    project_root = Path(__file__).parent.parent

    for clip in state['clips']:
        if clip.get('model') == model_name:
            video_path = Path(clip['output_path'])
            if not video_path.is_absolute():
                video_path = project_root / clip['output_path']
            if video_path.exists():
                return stream_video(video_path)

    for pattern in [f'{model_name}.mp4', f'{model_name}_h264.mp4']:
        for f in video_dir.glob(pattern):
            return stream_video(f)

    return jsonify({'error': 'Video file not found'}), 404


@app.route('/api/reprocess', methods=['POST'])
def reprocess():
    """Start re-processing with new random interval"""
    if processing_state['is_processing']:
        return jsonify({'error': 'Already processing'}), 409

    thread = threading.Thread(target=run_benchmark)
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started'})


@app.route('/api/processing-status')
def get_processing_status():
    """Get current processing status"""
    return jsonify(processing_state)


@app.route('/api/new-interval', methods=['POST'])
def new_interval():
    """Extract new random interval from pre-generated videos (fast) or fall back to regeneration"""
    import random

    project_root = Path(__file__).parent.parent
    full_processed_dir = project_root / 'outputs' / 'full_processed'
    raw_video = project_root / 'data' / 'raw' / 'clip1.mp4'

    # Check if pre-generated videos exist
    if full_processed_dir.exists():
        method_videos = list(full_processed_dir.glob('*.mp4'))
        if method_videos:
            try:
                # Import extract_clip function
                from benchmark_proper import extract_clip, get_full_video_duration, DURATION

                # Get duration of pre-generated video to calculate valid range
                first_video = method_videos[0]
                full_duration = get_full_video_duration(first_video)

                if full_duration > DURATION + 2:
                    # Pick random start time
                    max_start = full_duration - DURATION - 1
                    start_time = random.uniform(1, max_start)

                    # Extract clips for each method
                    outdir = state['video_dir']
                    for video_file in method_videos:
                        method = video_file.stem
                        output = outdir / f"{method}.mp4"
                        extract_clip(video_file, raw_video, start_time, DURATION, output)

                    # Reload metadata
                    state['metadata'] = load_video_metadata(state['video_dir'])
                    state['clips'] = state['metadata'].get('clips', [])
                    state['models'] = state['metadata'].get('models', [])

                    return jsonify({'status': 'done', 'start_time': round(start_time, 1)})
            except Exception as e:
                print(f"Fast interval extraction failed: {e}")
                # Fall through to regeneration

    # No pre-generated videos or extraction failed - fall back to regeneration
    if processing_state['is_processing']:
        return jsonify({'error': 'Already processing'}), 409

    thread = threading.Thread(target=run_benchmark)
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'fallback'})


@app.route('/api/available-clips')
def available_clips():
    """List available source video clips"""
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'data' / 'raw'

    clips = []
    if raw_dir.exists():
        clips = [f.stem for f in raw_dir.glob('*.mp4')]

    # Try to determine current clip from metadata
    current = state['metadata'].get('source_clip', 'clip1') if state['metadata'] else 'clip1'

    return jsonify({'clips': clips, 'current': current})


@app.route('/api/switch-clip', methods=['POST'])
def switch_clip():
    """Switch to a different source video clip"""
    data = request.get_json()
    clip_name = data.get('clip')

    if not clip_name:
        return jsonify({'error': 'No clip specified'}), 400

    project_root = Path(__file__).parent.parent
    clip_path = project_root / 'data' / 'raw' / f'{clip_name}.mp4'

    if not clip_path.exists():
        return jsonify({'error': f'Clip not found: {clip_name}'}), 404

    # Check for pre-generated videos for this clip
    full_processed_dir = project_root / 'outputs' / 'full_processed' / clip_name
    if full_processed_dir.exists() and list(full_processed_dir.glob('*.mp4')):
        # Has pre-generated videos - extract a random interval
        import random
        try:
            from benchmark_proper import extract_clip, get_full_video_duration, DURATION

            method_videos = list(full_processed_dir.glob('*.mp4'))
            first_video = method_videos[0]
            full_duration = get_full_video_duration(first_video)

            if full_duration > DURATION + 2:
                max_start = full_duration - DURATION - 1
                start_time = random.uniform(1, max_start)

                outdir = state['video_dir']
                for video_file in method_videos:
                    method = video_file.stem
                    output = outdir / f"{method}.mp4"
                    extract_clip(video_file, clip_path, start_time, DURATION, output)

                # Update metadata with new source
                state['metadata'] = load_video_metadata(state['video_dir'])
                state['metadata']['source_clip'] = clip_name
                state['clips'] = state['metadata'].get('clips', [])
                state['models'] = state['metadata'].get('models', [])

                return jsonify({'status': 'done', 'clip': clip_name})
        except Exception as e:
            print(f"Fast clip switch failed: {e}")

    # No pre-generated videos - would need full regeneration
    return jsonify({
        'error': f'No pre-generated videos for {clip_name}. Use Regenerate to process this clip.',
        'needs_regeneration': True
    }), 400


def main():
    parser = argparse.ArgumentParser(description='Video-based Blind Study GUI')
    parser.add_argument('--video-dir', type=str, default='outputs/benchmark')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = project_root / args.video_dir

    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        sys.exit(1)

    state['video_dir'] = video_dir
    state['metadata'] = load_video_metadata(video_dir)
    state['clips'] = state['metadata'].get('clips', [])
    state['models'] = state['metadata'].get('models', [])

    if not state['clips']:
        print(f"Error: No video clips found in {video_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("VFI+SR Video Blind Study")
    print(f"{'='*60}")
    print(f"Video directory: {video_dir}")
    print(f"Models: {state['models']}")
    print(f"Clips: {len(state['clips'])}")
    print(f"\nOpen in browser: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
