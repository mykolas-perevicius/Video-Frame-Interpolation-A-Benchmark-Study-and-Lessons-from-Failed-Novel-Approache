#!/usr/bin/env python3
"""
Blind Study GUI for VFI+SR Model Comparison

Two phases:
1. Learning Phase: View comparisons with model labels shown
2. Quiz Phase: Guess which model produced each result

Usage:
    python gui/blind_study_app.py
    python gui/blind_study_app.py --data-dir data/processed --results outputs/benchmarks/benchmark_results.json
"""

import argparse
import json
import random
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk, messagebox
from typing import List, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ComparisonPair:
    """A pair of frames for comparison"""
    frame_idx: int
    input_frame_path: Path
    gt_frame_path: Path
    model_a_name: str
    model_b_name: str
    model_a_frame: Optional[np.ndarray] = None
    model_b_frame: Optional[np.ndarray] = None


class BlindStudyApp:
    """
    GUI Application for blind comparison study.

    Features:
    - Learning phase: See comparisons with labels
    - Quiz phase: Guess which model is which
    - Progress tracking
    - Score display
    """

    def __init__(self, data_dir: Path, results_path: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.results_path = results_path

        # Load available clips
        self.clips = self._load_clips()
        if not self.clips:
            raise ValueError(f"No processed clips found in {data_dir}")

        # Load models from results if available
        self.models = self._load_models()

        # Study state
        self.current_clip = list(self.clips.keys())[0]
        self.current_pair_idx = 0
        self.pairs: List[ComparisonPair] = []
        self.quiz_answers: List[Dict] = []
        self.phase = "learning"  # "learning" or "quiz"
        self.quiz_score = 0
        self.quiz_total = 0

        # Create main window
        self.root = tk.Tk()
        self.root.title("VFI+SR Blind Study")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')

        # Create UI
        self._create_ui()

        # Generate initial pairs
        self._generate_pairs()
        self._show_current_pair()

    def _load_clips(self) -> Dict[str, Path]:
        """Load available processed clips"""
        clips = {}
        for clip_dir in self.data_dir.iterdir():
            if clip_dir.is_dir() and (clip_dir / 'metadata.json').exists():
                clips[clip_dir.name] = clip_dir
        return clips

    def _load_models(self) -> List[str]:
        """Load available models from results"""
        if self.results_path and self.results_path.exists():
            with open(self.results_path) as f:
                results = json.load(f)
            # Get models from first clip
            first_clip = list(results.keys())[0]
            return list(results[first_clip].keys())
        # Default models
        return ['bicubic', 'lanczos', 'optical_flow']

    def _create_ui(self):
        """Create the user interface"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white', font=('Helvetica', 11))
        style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Score.TLabel', font=('Helvetica', 14), foreground='#00ff00')
        style.configure('TButton', font=('Helvetica', 11))
        style.configure('Big.TButton', font=('Helvetica', 14, 'bold'), padding=10)

        # Main container
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Header frame
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))

        # Phase indicator
        self.phase_label = ttk.Label(
            self.header_frame,
            text="LEARNING PHASE",
            style='Header.TLabel'
        )
        self.phase_label.pack(side=tk.LEFT)

        # Score display (for quiz phase)
        self.score_label = ttk.Label(
            self.header_frame,
            text="",
            style='Score.TLabel'
        )
        self.score_label.pack(side=tk.RIGHT)

        # Progress indicator
        self.progress_label = ttk.Label(
            self.header_frame,
            text="Pair 1 / 10"
        )
        self.progress_label.pack(side=tk.RIGHT, padx=20)

        # Image comparison frame
        self.comparison_frame = ttk.Frame(self.main_frame)
        self.comparison_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left panel (Model A / Option A)
        self.left_panel = ttk.Frame(self.comparison_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.left_label = ttk.Label(self.left_panel, text="Model A: ???", style='Header.TLabel')
        self.left_label.pack(pady=5)

        self.left_canvas = tk.Canvas(self.left_panel, bg='#1a1a1a', highlightthickness=2, highlightbackground='#444')
        self.left_canvas.pack(fill=tk.BOTH, expand=True)

        # Right panel (Model B / Option B)
        self.right_panel = ttk.Frame(self.comparison_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.right_label = ttk.Label(self.right_panel, text="Model B: ???", style='Header.TLabel')
        self.right_label.pack(pady=5)

        self.right_canvas = tk.Canvas(self.right_panel, bg='#1a1a1a', highlightthickness=2, highlightbackground='#444')
        self.right_canvas.pack(fill=tk.BOTH, expand=True)

        # Ground truth panel (smaller, at bottom)
        self.gt_frame = ttk.Frame(self.main_frame)
        self.gt_frame.pack(fill=tk.X, pady=10)

        ttk.Label(self.gt_frame, text="Ground Truth (Reference)", style='Header.TLabel').pack()
        self.gt_canvas = tk.Canvas(self.gt_frame, bg='#1a1a1a', height=200, highlightthickness=1, highlightbackground='#444')
        self.gt_canvas.pack(fill=tk.X, pady=5)

        # Control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)

        # Navigation buttons
        self.nav_frame = ttk.Frame(self.control_frame)
        self.nav_frame.pack(side=tk.LEFT)

        self.prev_btn = ttk.Button(self.nav_frame, text="< Previous", command=self._prev_pair)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(self.nav_frame, text="Next >", command=self._next_pair)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # Quiz buttons (hidden in learning phase)
        self.quiz_frame = ttk.Frame(self.control_frame)
        self.quiz_frame.pack(side=tk.LEFT, padx=50)

        self.guess_a_btn = ttk.Button(
            self.quiz_frame,
            text="Left is Better",
            command=lambda: self._submit_guess('left'),
            style='Big.TButton'
        )
        self.guess_a_btn.pack(side=tk.LEFT, padx=10)

        self.guess_b_btn = ttk.Button(
            self.quiz_frame,
            text="Right is Better",
            command=lambda: self._submit_guess('right'),
            style='Big.TButton'
        )
        self.guess_b_btn.pack(side=tk.LEFT, padx=10)

        self.guess_same_btn = ttk.Button(
            self.quiz_frame,
            text="Same / Can't Tell",
            command=lambda: self._submit_guess('same')
        )
        self.guess_same_btn.pack(side=tk.LEFT, padx=10)

        # Initially hide quiz buttons
        self.quiz_frame.pack_forget()

        # Phase switch button
        self.phase_frame = ttk.Frame(self.control_frame)
        self.phase_frame.pack(side=tk.RIGHT)

        self.switch_phase_btn = ttk.Button(
            self.phase_frame,
            text="Start Quiz Phase",
            command=self._toggle_phase,
            style='Big.TButton'
        )
        self.switch_phase_btn.pack(padx=10)

        # Clip selector
        self.clip_frame = ttk.Frame(self.control_frame)
        self.clip_frame.pack(side=tk.RIGHT, padx=20)

        ttk.Label(self.clip_frame, text="Clip:").pack(side=tk.LEFT)
        self.clip_var = tk.StringVar(value=self.current_clip)
        self.clip_selector = ttk.Combobox(
            self.clip_frame,
            textvariable=self.clip_var,
            values=list(self.clips.keys()),
            state='readonly',
            width=20
        )
        self.clip_selector.pack(side=tk.LEFT, padx=5)
        self.clip_selector.bind('<<ComboboxSelected>>', self._on_clip_change)

        # Bind canvas resize
        self.left_canvas.bind('<Configure>', lambda e: self._show_current_pair())
        self.right_canvas.bind('<Configure>', lambda e: self._show_current_pair())

        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self._prev_pair())
        self.root.bind('<Right>', lambda e: self._next_pair())
        self.root.bind('a', lambda e: self._submit_guess('left') if self.phase == 'quiz' else None)
        self.root.bind('l', lambda e: self._submit_guess('right') if self.phase == 'quiz' else None)
        self.root.bind('s', lambda e: self._submit_guess('same') if self.phase == 'quiz' else None)
        self.root.bind('<space>', lambda e: self._toggle_phase())

    def _generate_pairs(self, num_pairs: int = 10):
        """Generate comparison pairs for the current clip"""
        clip_dir = self.clips[self.current_clip]

        # Load triplets
        with open(clip_dir / 'triplets.json') as f:
            triplets = json.load(f)

        # Select random triplets
        selected = random.sample(triplets, min(num_pairs, len(triplets)))

        self.pairs = []
        available_models = self.models[:3]  # Use first 3 models for comparisons

        for triplet in selected:
            # Random pair of models
            if len(available_models) >= 2:
                model_a, model_b = random.sample(available_models, 2)
            else:
                model_a, model_b = available_models[0], available_models[0]

            # Get frame paths
            input_dir = clip_dir / 'input_1080p30' / 'frames'
            gt_dir = clip_dir / 'ground_truth' / 'frames'

            pair = ComparisonPair(
                frame_idx=triplet['frame0_idx'],
                input_frame_path=input_dir / f"frame_{triplet['frame0_idx']:06d}.png",
                gt_frame_path=gt_dir / f"frame_{triplet['gt_indices'][0]:06d}.png",
                model_a_name=model_a,
                model_b_name=model_b,
            )

            # Generate model outputs
            pair.model_a_frame = self._apply_model(pair.input_frame_path, model_a)
            pair.model_b_frame = self._apply_model(pair.input_frame_path, model_b)

            self.pairs.append(pair)

        self.current_pair_idx = 0

    def _apply_model(self, frame_path: Path, model_name: str) -> np.ndarray:
        """Apply a model to generate output frame"""
        frame = cv2.imread(str(frame_path))
        if frame is None:
            # Return placeholder if frame not found
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply model-specific processing
        h, w = frame.shape[:2]
        scale = 1.333  # 1080p -> 1440p
        new_h, new_w = int(h * scale), int(w * scale)

        if model_name == 'bicubic':
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif model_name == 'lanczos':
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        elif model_name == 'optical_flow':
            # For simplicity, use Lanczos with slight blur to simulate
            result = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            return cv2.GaussianBlur(result, (3, 3), 0.5)
        else:
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _show_current_pair(self):
        """Display the current comparison pair"""
        if not self.pairs:
            return

        pair = self.pairs[self.current_pair_idx]

        # Update labels based on phase
        if self.phase == "learning":
            self.left_label.config(text=f"Model A: {pair.model_a_name.upper()}")
            self.right_label.config(text=f"Model B: {pair.model_b_name.upper()}")
        else:
            self.left_label.config(text="Option A (which model?)")
            self.right_label.config(text="Option B (which model?)")

        # Update progress
        self.progress_label.config(text=f"Pair {self.current_pair_idx + 1} / {len(self.pairs)}")

        # Display frames
        self._display_frame(pair.model_a_frame, self.left_canvas)
        self._display_frame(pair.model_b_frame, self.right_canvas)

        # Display ground truth
        gt_frame = cv2.imread(str(pair.gt_frame_path))
        if gt_frame is not None:
            gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB)
            self._display_frame(gt_frame, self.gt_canvas, max_height=180)

    def _display_frame(self, frame: np.ndarray, canvas: tk.Canvas, max_height: int = None):
        """Display a frame on a canvas, scaled to fit"""
        if frame is None:
            return

        canvas.update()
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height() if max_height is None else min(canvas.winfo_height(), max_height)

        if canvas_w < 10 or canvas_h < 10:
            return

        # Scale frame to fit canvas
        h, w = frame.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        if new_w < 1 or new_h < 1:
            return

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to PhotoImage
        img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(img)

        # Store reference to prevent garbage collection
        canvas.photo = photo

        # Center on canvas
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2

        canvas.delete('all')
        canvas.create_image(x, y, anchor=tk.NW, image=photo)

    def _prev_pair(self):
        """Go to previous pair"""
        if self.current_pair_idx > 0:
            self.current_pair_idx -= 1
            self._show_current_pair()

    def _next_pair(self):
        """Go to next pair"""
        if self.current_pair_idx < len(self.pairs) - 1:
            self.current_pair_idx += 1
            self._show_current_pair()

    def _toggle_phase(self):
        """Toggle between learning and quiz phases"""
        if self.phase == "learning":
            self.phase = "quiz"
            self.phase_label.config(text="QUIZ PHASE - Guess the better model!")
            self.switch_phase_btn.config(text="Back to Learning")
            self.quiz_frame.pack(side=tk.LEFT, padx=50)
            self.quiz_score = 0
            self.quiz_total = 0
            self._update_score()

            # Shuffle pairs for quiz
            random.shuffle(self.pairs)

            # Randomize left/right for each pair
            for pair in self.pairs:
                if random.random() > 0.5:
                    pair.model_a_frame, pair.model_b_frame = pair.model_b_frame, pair.model_a_frame
                    pair.model_a_name, pair.model_b_name = pair.model_b_name, pair.model_a_name

            self.current_pair_idx = 0
        else:
            self.phase = "learning"
            self.phase_label.config(text="LEARNING PHASE")
            self.switch_phase_btn.config(text="Start Quiz Phase")
            self.quiz_frame.pack_forget()
            self.score_label.config(text="")
            self._generate_pairs()

        self._show_current_pair()

    def _submit_guess(self, choice: str):
        """Submit a guess in quiz phase"""
        if self.phase != "quiz":
            return

        pair = self.pairs[self.current_pair_idx]

        # Determine which model is "better" (lower LPIPS = better perceptual quality)
        # For simplicity, use a ranking: lanczos > bicubic > optical_flow
        model_ranking = {'lanczos': 3, 'bicubic': 2, 'optical_flow': 1}

        rank_a = model_ranking.get(pair.model_a_name, 0)
        rank_b = model_ranking.get(pair.model_b_name, 0)

        if rank_a > rank_b:
            correct = 'left'
        elif rank_b > rank_a:
            correct = 'right'
        else:
            correct = 'same'

        # Record answer
        self.quiz_answers.append({
            'pair_idx': self.current_pair_idx,
            'model_a': pair.model_a_name,
            'model_b': pair.model_b_name,
            'user_choice': choice,
            'correct_choice': correct,
            'is_correct': choice == correct or (correct == 'same' and choice == 'same')
        })

        self.quiz_total += 1

        # Check if correct
        if choice == correct or (choice == 'same' and abs(rank_a - rank_b) == 0):
            self.quiz_score += 1

        self._update_score()

        # Show feedback briefly
        self.left_label.config(text=f"Left: {pair.model_a_name.upper()}")
        self.right_label.config(text=f"Right: {pair.model_b_name.upper()}")

        # Highlight correct answer
        if correct == 'left':
            self.left_canvas.config(highlightbackground='#00ff00')
            self.right_canvas.config(highlightbackground='#444')
        elif correct == 'right':
            self.left_canvas.config(highlightbackground='#444')
            self.right_canvas.config(highlightbackground='#00ff00')
        else:
            self.left_canvas.config(highlightbackground='#ffff00')
            self.right_canvas.config(highlightbackground='#ffff00')

        # Auto-advance after delay
        self.root.after(1500, self._advance_quiz)

    def _advance_quiz(self):
        """Advance to next quiz pair or show results"""
        # Reset highlight
        self.left_canvas.config(highlightbackground='#444')
        self.right_canvas.config(highlightbackground='#444')

        if self.current_pair_idx < len(self.pairs) - 1:
            self.current_pair_idx += 1
            self._show_current_pair()
        else:
            self._show_quiz_results()

    def _update_score(self):
        """Update the score display"""
        if self.quiz_total > 0:
            pct = (self.quiz_score / self.quiz_total) * 100
            self.score_label.config(text=f"Score: {self.quiz_score}/{self.quiz_total} ({pct:.0f}%)")
        else:
            self.score_label.config(text="Score: 0/0")

    def _show_quiz_results(self):
        """Show final quiz results"""
        pct = (self.quiz_score / self.quiz_total) * 100 if self.quiz_total > 0 else 0

        msg = f"""Quiz Complete!

Your Score: {self.quiz_score} / {self.quiz_total} ({pct:.0f}%)

"""
        if pct >= 80:
            msg += "Excellent! You have a great eye for quality differences!"
        elif pct >= 60:
            msg += "Good job! You can distinguish most quality differences."
        elif pct >= 40:
            msg += "Not bad! The differences can be subtle."
        else:
            msg += "The models produce similar results - it's hard to tell them apart!"

        messagebox.showinfo("Quiz Results", msg)

        # Reset to learning phase
        self._toggle_phase()

    def _on_clip_change(self, event):
        """Handle clip selection change"""
        self.current_clip = self.clip_var.get()
        self._generate_pairs()
        self._show_current_pair()

    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Blind Study GUI for VFI+SR comparison')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing processed clips')
    parser.add_argument('--results', type=str, default='outputs/benchmarks/benchmark_results.json',
                        help='Benchmark results JSON file')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_path = Path(args.results) if args.results else None

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    app = BlindStudyApp(data_dir, results_path)
    app.run()


if __name__ == '__main__':
    main()
