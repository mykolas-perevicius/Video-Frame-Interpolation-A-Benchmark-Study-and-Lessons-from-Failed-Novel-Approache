#!/usr/bin/env python3
"""
Web-based Blind Study GUI for VFI+SR Model Comparison

Two phases:
1. Learning Phase: View comparisons with model labels shown
2. Quiz Phase: Guess which model produced each result

Usage:
    python gui/web_app.py
    python gui/web_app.py --port 5000

Then open http://localhost:5000 in your browser
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from flask import Flask, render_template_string, jsonify, request, send_file

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)

# Global state
state = {
    'data_dir': None,
    'clips': {},
    'models': ['control', 'degraded', 'lanczos_blend', 'lanczos_blend_edge',
               'optical_flow_basic', 'optical_flow_edge', 'bicubic_blend', 'bicubic_blend_edge',
               'rife_default', 'adaptive_conservative', 'adaptive_default', 'adaptive_aggressive',
               'uafi_default', 'ughi_default', 'mcar_default', 'mcar_aggressive'],
    'current_clip': None,
    'pairs': [],
    'current_pair_idx': 0,
    'phase': 'learning',
    'quiz_score': 0,
    'quiz_total': 0,
    'quiz_answers': []
}


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VFI+SR Blind Study</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .phase-label {
            font-size: 1.5em;
            font-weight: bold;
            color: #00d4ff;
        }
        .score {
            font-size: 1.3em;
            color: #00ff88;
        }
        .progress {
            font-size: 1.1em;
            color: #aaa;
        }
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .panel h3 {
            margin-bottom: 10px;
            font-size: 1.2em;
            color: #00d4ff;
        }
        .panel img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 3px solid #333;
            transition: border-color 0.3s;
        }
        .panel.correct img {
            border-color: #00ff88;
        }
        .panel.incorrect img {
            border-color: #ff4444;
        }
        .gt-panel {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin-bottom: 20px;
        }
        .gt-panel h3 {
            margin-bottom: 10px;
            color: #ffd700;
        }
        .gt-panel img {
            max-width: 50%;
            height: auto;
            border-radius: 8px;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        button {
            padding: 12px 25px;
            font-size: 1em;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        .nav-btn {
            background: #444;
            color: #fff;
        }
        .nav-btn:hover {
            background: #555;
        }
        .quiz-btn {
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #000;
            font-size: 1.1em;
            padding: 15px 30px;
        }
        .quiz-btn:hover {
            transform: scale(1.05);
        }
        .quiz-btn.left {
            background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        }
        .quiz-btn.right {
            background: linear-gradient(135deg, #4ecdc4, #44a08d);
        }
        .quiz-btn.same {
            background: linear-gradient(135deg, #f9d423, #ff4e50);
        }
        .phase-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #fff;
        }
        .phase-btn:hover {
            transform: scale(1.05);
        }
        .clip-selector {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
        }
        select {
            padding: 10px 15px;
            font-size: 1em;
            border-radius: 8px;
            border: none;
            background: #333;
            color: #fff;
        }
        .feedback {
            text-align: center;
            font-size: 1.3em;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }
        .feedback.correct {
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
            display: block;
        }
        .feedback.incorrect {
            background: rgba(255, 68, 68, 0.2);
            color: #ff4444;
            display: block;
        }
        .instructions {
            background: rgba(255,255,255,0.1);
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .instructions h4 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        .instructions ul {
            margin-left: 20px;
        }
        .results-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .results-content {
            background: #1a1a2e;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            max-width: 500px;
        }
        .results-content h2 {
            color: #00d4ff;
            margin-bottom: 20px;
        }
        .results-score {
            font-size: 3em;
            color: #00ff88;
            margin: 20px 0;
        }
        .keyboard-hints {
            margin-top: 20px;
            font-size: 0.9em;
            color: #888;
        }
        .keyboard-hints kbd {
            background: #333;
            padding: 3px 8px;
            border-radius: 4px;
            margin: 0 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="phase-label" id="phaseLabel">LEARNING PHASE</div>
            <div class="progress" id="progress">Pair 1 / 10</div>
            <div class="score" id="score"></div>
        </header>

        <div class="instructions" id="instructions">
            <h4>How to use:</h4>
            <ul>
                <li><strong>Learning Phase:</strong> Study the differences between models. Labels are shown.</li>
                <li><strong>Quiz Phase:</strong> Labels hidden - guess which output looks better!</li>
                <li>Use keyboard shortcuts: <kbd>←</kbd> <kbd>→</kbd> to navigate, <kbd>A</kbd> <kbd>L</kbd> <kbd>S</kbd> to answer</li>
            </ul>
        </div>

        <div class="feedback" id="feedback"></div>

        <div class="comparison">
            <div class="panel" id="leftPanel">
                <h3 id="leftLabel">Model A: ???</h3>
                <img id="leftImage" src="" alt="Model A output">
            </div>
            <div class="panel" id="rightPanel">
                <h3 id="rightLabel">Model B: ???</h3>
                <img id="rightImage" src="" alt="Model B output">
            </div>
        </div>

        <div class="gt-panel">
            <h3>Ground Truth (Reference)</h3>
            <img id="gtImage" src="" alt="Ground truth">
        </div>

        <div class="controls">
            <button class="nav-btn" onclick="prevPair()">← Previous</button>
            <button class="nav-btn" onclick="nextPair()">Next →</button>
        </div>

        <div class="controls" id="quizControls" style="display: none;">
            <button class="quiz-btn left" onclick="submitGuess('left')">Left is Better (A)</button>
            <button class="quiz-btn same" onclick="submitGuess('same')">Same / Can't Tell (S)</button>
            <button class="quiz-btn right" onclick="submitGuess('right')">Right is Better (L)</button>
        </div>

        <div class="controls">
            <button class="phase-btn" id="phaseBtn" onclick="togglePhase()">Start Quiz Phase</button>
        </div>

        <div class="clip-selector">
            <label for="clipSelect">Select Clip:</label>
            <select id="clipSelect" onchange="changeClip()"></select>
        </div>

        <div class="keyboard-hints">
            Keyboard: <kbd>←</kbd>/<kbd>→</kbd> Navigate | <kbd>A</kbd> Left Better | <kbd>L</kbd> Right Better | <kbd>S</kbd> Same | <kbd>Space</kbd> Toggle Phase
        </div>
    </div>

    <div class="results-modal" id="resultsModal">
        <div class="results-content">
            <h2>Quiz Complete!</h2>
            <div class="results-score" id="finalScore">0/0</div>
            <p id="resultMessage"></p>
            <button class="phase-btn" onclick="closeResults()" style="margin-top: 20px;">Continue Learning</button>
        </div>
    </div>

    <script>
        let state = {
            phase: 'learning',
            currentPairIdx: 0,
            pairs: [],
            quizScore: 0,
            quizTotal: 0
        };

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadClips();
            loadPairs();

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowLeft') prevPair();
                else if (e.key === 'ArrowRight') nextPair();
                else if (e.key === 'a' || e.key === 'A') { if (state.phase === 'quiz') submitGuess('left'); }
                else if (e.key === 'l' || e.key === 'L') { if (state.phase === 'quiz') submitGuess('right'); }
                else if (e.key === 's' || e.key === 'S') { if (state.phase === 'quiz') submitGuess('same'); }
                else if (e.key === ' ') { e.preventDefault(); togglePhase(); }
            });
        });

        async function loadClips() {
            const response = await fetch('/api/clips');
            const clips = await response.json();
            const select = document.getElementById('clipSelect');
            select.innerHTML = clips.map(c => `<option value="${c}">${c}</option>`).join('');
        }

        async function loadPairs() {
            const response = await fetch('/api/pairs');
            state.pairs = await response.json();
            state.currentPairIdx = 0;
            updateDisplay();
        }

        function updateDisplay() {
            if (state.pairs.length === 0) return;

            const pair = state.pairs[state.currentPairIdx];

            // Update labels
            if (state.phase === 'learning') {
                document.getElementById('leftLabel').textContent = `Model A: ${pair.model_a.toUpperCase()}`;
                document.getElementById('rightLabel').textContent = `Model B: ${pair.model_b.toUpperCase()}`;
            } else {
                document.getElementById('leftLabel').textContent = 'Option A (which is better?)';
                document.getElementById('rightLabel').textContent = 'Option B (which is better?)';
            }

            // Update images
            document.getElementById('leftImage').src = `/api/frame/${state.currentPairIdx}/left`;
            document.getElementById('rightImage').src = `/api/frame/${state.currentPairIdx}/right`;
            document.getElementById('gtImage').src = `/api/frame/${state.currentPairIdx}/gt`;

            // Update progress
            document.getElementById('progress').textContent = `Pair ${state.currentPairIdx + 1} / ${state.pairs.length}`;

            // Reset panel styling
            document.getElementById('leftPanel').className = 'panel';
            document.getElementById('rightPanel').className = 'panel';
            document.getElementById('feedback').className = 'feedback';
        }

        function prevPair() {
            if (state.currentPairIdx > 0) {
                state.currentPairIdx--;
                updateDisplay();
            }
        }

        function nextPair() {
            if (state.currentPairIdx < state.pairs.length - 1) {
                state.currentPairIdx++;
                updateDisplay();
            }
        }

        async function togglePhase() {
            if (state.phase === 'learning') {
                state.phase = 'quiz';
                document.getElementById('phaseLabel').textContent = 'QUIZ PHASE - Pick the better one!';
                document.getElementById('phaseBtn').textContent = 'Back to Learning';
                document.getElementById('quizControls').style.display = 'flex';
                document.getElementById('instructions').style.display = 'none';
                state.quizScore = 0;
                state.quizTotal = 0;
                updateScore();

                // Shuffle pairs via API
                const response = await fetch('/api/shuffle', { method: 'POST' });
                state.pairs = await response.json();
                state.currentPairIdx = 0;
            } else {
                state.phase = 'learning';
                document.getElementById('phaseLabel').textContent = 'LEARNING PHASE';
                document.getElementById('phaseBtn').textContent = 'Start Quiz Phase';
                document.getElementById('quizControls').style.display = 'none';
                document.getElementById('instructions').style.display = 'block';
                document.getElementById('score').textContent = '';

                // Reload fresh pairs
                await loadPairs();
            }
            updateDisplay();
        }

        async function submitGuess(choice) {
            if (state.phase !== 'quiz') return;

            const response = await fetch('/api/guess', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    pair_idx: state.currentPairIdx,
                    choice: choice
                })
            });

            const result = await response.json();
            state.quizScore = result.score;
            state.quizTotal = result.total;
            updateScore();

            // Show feedback
            const feedback = document.getElementById('feedback');
            const pair = state.pairs[state.currentPairIdx];

            if (result.correct) {
                feedback.textContent = '✓ Correct!';
                feedback.className = 'feedback correct';
            } else {
                feedback.textContent = `✗ Wrong! Left: ${pair.model_a}, Right: ${pair.model_b}`;
                feedback.className = 'feedback incorrect';
            }

            // Highlight panels
            document.getElementById('leftPanel').className = 'panel ' + (result.correct_choice === 'left' ? 'correct' : '');
            document.getElementById('rightPanel').className = 'panel ' + (result.correct_choice === 'right' ? 'correct' : '');

            // Show labels
            document.getElementById('leftLabel').textContent = `Left: ${pair.model_a.toUpperCase()}`;
            document.getElementById('rightLabel').textContent = `Right: ${pair.model_b.toUpperCase()}`;

            // Auto-advance
            setTimeout(() => {
                if (state.currentPairIdx < state.pairs.length - 1) {
                    state.currentPairIdx++;
                    updateDisplay();
                } else {
                    showResults();
                }
            }, 1500);
        }

        function updateScore() {
            const pct = state.quizTotal > 0 ? Math.round((state.quizScore / state.quizTotal) * 100) : 0;
            document.getElementById('score').textContent = `Score: ${state.quizScore}/${state.quizTotal} (${pct}%)`;
        }

        function showResults() {
            const pct = state.quizTotal > 0 ? Math.round((state.quizScore / state.quizTotal) * 100) : 0;
            document.getElementById('finalScore').textContent = `${state.quizScore}/${state.quizTotal} (${pct}%)`;

            let msg = '';
            if (pct >= 80) msg = 'Excellent! You have a great eye for quality differences!';
            else if (pct >= 60) msg = 'Good job! You can distinguish most quality differences.';
            else if (pct >= 40) msg = 'Not bad! The differences can be subtle.';
            else msg = 'The models produce similar results - hard to tell them apart!';

            document.getElementById('resultMessage').textContent = msg;
            document.getElementById('resultsModal').style.display = 'flex';
        }

        function closeResults() {
            document.getElementById('resultsModal').style.display = 'none';
            togglePhase();
        }

        async function changeClip() {
            const clip = document.getElementById('clipSelect').value;
            await fetch('/api/clip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clip: clip })
            });
            await loadPairs();
        }
    </script>
</body>
</html>
'''


def load_clips(data_dir: Path) -> Dict[str, Path]:
    """Load available processed clips"""
    clips = {}
    for clip_dir in data_dir.iterdir():
        if clip_dir.is_dir() and (clip_dir / 'metadata.json').exists():
            clips[clip_dir.name] = clip_dir
    return clips


def generate_pairs(clip_dir: Path, num_pairs: int = 10) -> List[Dict]:
    """Generate comparison pairs for a clip"""
    with open(clip_dir / 'triplets.json') as f:
        triplets = json.load(f)

    selected = random.sample(triplets, min(num_pairs, len(triplets)))
    pairs = []

    models = state['models'][:3]

    for triplet in selected:
        if len(models) >= 2:
            model_a, model_b = random.sample(models, 2)
        else:
            model_a = model_b = models[0]

        # Use correct key names from triplet structure
        frame_idx = triplet.get('input_idx_0', triplet.get('frame0_idx', 0))
        gt_indices = triplet.get('gt_intermediate_indices', triplet.get('gt_indices', []))
        gt_idx = gt_indices[0] if gt_indices else frame_idx

        pairs.append({
            'frame_idx': frame_idx,
            'gt_idx': gt_idx,
            'model_a': model_a,
            'model_b': model_b,
            'swapped': False
        })

    return pairs


def apply_model(frame: np.ndarray, model_name: str) -> np.ndarray:
    """Apply a model to generate output frame"""
    h, w = frame.shape[:2]
    scale = 1.333
    new_h, new_w = int(h * scale), int(w * scale)

    if model_name == 'bicubic':
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif model_name == 'lanczos':
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    elif model_name == 'optical_flow':
        result = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return cv2.GaussianBlur(result, (3, 3), 0.5)
    else:
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/clips')
def get_clips():
    return jsonify(list(state['clips'].keys()))


@app.route('/api/clip', methods=['POST'])
def set_clip():
    data = request.json
    state['current_clip'] = data['clip']
    clip_dir = state['clips'][state['current_clip']]
    state['pairs'] = generate_pairs(clip_dir)
    state['current_pair_idx'] = 0
    return jsonify({'status': 'ok'})


@app.route('/api/pairs')
def get_pairs():
    if not state['pairs']:
        clip_dir = state['clips'][state['current_clip']]
        state['pairs'] = generate_pairs(clip_dir)
    return jsonify(state['pairs'])


@app.route('/api/shuffle', methods=['POST'])
def shuffle_pairs():
    """Shuffle pairs and randomize left/right for quiz"""
    random.shuffle(state['pairs'])
    for pair in state['pairs']:
        if random.random() > 0.5:
            pair['model_a'], pair['model_b'] = pair['model_b'], pair['model_a']
            pair['swapped'] = not pair.get('swapped', False)
    state['quiz_score'] = 0
    state['quiz_total'] = 0
    return jsonify(state['pairs'])


@app.route('/api/frame/<int:pair_idx>/<position>')
def get_frame(pair_idx: int, position: str):
    """Get a frame image"""
    if pair_idx >= len(state['pairs']):
        return '', 404

    pair = state['pairs'][pair_idx]
    clip_dir = state['clips'][state['current_clip']]
    input_dir = clip_dir / 'input_1080p30' / 'frames'
    gt_dir = clip_dir / 'ground_truth' / 'frames'

    if position == 'gt':
        frame_path = gt_dir / f"frame_{pair['gt_idx']:06d}.png"
        if frame_path.exists():
            frame = cv2.imread(str(frame_path))
        else:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    else:
        input_path = input_dir / f"frame_{pair['frame_idx']:06d}.png"
        if input_path.exists():
            input_frame = cv2.imread(str(input_path))
        else:
            input_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        model = pair['model_a'] if position == 'left' else pair['model_b']
        frame = apply_model(input_frame, model)

    # Convert to JPEG and return
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')


@app.route('/api/guess', methods=['POST'])
def submit_guess():
    """Submit a quiz guess"""
    data = request.json
    pair_idx = data['pair_idx']
    choice = data['choice']

    if pair_idx >= len(state['pairs']):
        return jsonify({'error': 'Invalid pair index'}), 400

    pair = state['pairs'][pair_idx]

    # Determine correct answer based on model quality ranking (from experiment results)
    model_ranking = {
        'rife_default': 12,
        'mcar_default': 11,
        'mcar_aggressive': 11,
        'adaptive_default': 10,
        'adaptive_conservative': 10,
        'adaptive_aggressive': 10,
        'lanczos_blend': 9,
        'optical_flow_basic': 9,
        'lanczos_blend_edge': 8,
        'optical_flow_edge': 8,
        'bicubic_blend_edge': 7,
        'uafi_default': 6,
        'ughi_default': 6,
        'bicubic_blend': 5,
        'lanczos': 4,  # legacy
        'bicubic': 3,  # legacy
        'optical_flow': 2,  # legacy
        'degraded': 1,
        'control': 0,  # Reference, not compared
    }
    rank_a = model_ranking.get(pair['model_a'], 0)
    rank_b = model_ranking.get(pair['model_b'], 0)

    if rank_a > rank_b:
        correct = 'left'
    elif rank_b > rank_a:
        correct = 'right'
    else:
        correct = 'same'

    state['quiz_total'] += 1
    is_correct = (choice == correct) or (choice == 'same' and correct == 'same')

    if is_correct:
        state['quiz_score'] += 1

    return jsonify({
        'correct': is_correct,
        'correct_choice': correct,
        'score': state['quiz_score'],
        'total': state['quiz_total'],
        'model_a': pair['model_a'],
        'model_b': pair['model_b']
    })


def main():
    parser = argparse.ArgumentParser(description='Web-based Blind Study GUI')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing processed clips')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Initialize state
    state['data_dir'] = data_dir
    state['clips'] = load_clips(data_dir)

    if not state['clips']:
        print(f"Error: No processed clips found in {data_dir}")
        sys.exit(1)

    state['current_clip'] = list(state['clips'].keys())[0]
    clip_dir = state['clips'][state['current_clip']]
    state['pairs'] = generate_pairs(clip_dir)

    print(f"\n{'='*50}")
    print("VFI+SR Blind Study Web App")
    print(f"{'='*50}")
    print(f"Data directory: {data_dir}")
    print(f"Available clips: {list(state['clips'].keys())}")
    print(f"Models: {state['models']}")
    print(f"\nOpen in browser: http://localhost:{args.port}")
    print(f"{'='*50}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
