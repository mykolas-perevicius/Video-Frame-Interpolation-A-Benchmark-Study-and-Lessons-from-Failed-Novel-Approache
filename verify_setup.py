#!/usr/bin/env python3
"""
verify_setup.py - Verify that the environment is correctly set up

Run after setup_wsl.sh to confirm everything is working.

Usage:
    python verify_setup.py
"""

import sys
import subprocess
from pathlib import Path


def check(name: str, condition: bool, detail: str = ""):
    """Print check result"""
    status = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"
    reset = "\033[0m"
    detail_str = f" ({detail})" if detail else ""
    print(f"  {color}{status}{reset} {name}{detail_str}")
    return condition


def main():
    print("=" * 50)
    print(" ENVIRONMENT VERIFICATION")
    print("=" * 50)
    
    all_passed = True
    
    # Python version
    print("\n[Python]")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    all_passed &= check("Python 3.10+", sys.version_info >= (3, 10), py_version)
    
    # PyTorch + CUDA
    print("\n[PyTorch + CUDA]")
    try:
        import torch
        all_passed &= check("PyTorch installed", True, torch.__version__)
        all_passed &= check("CUDA available", torch.cuda.is_available())
        if torch.cuda.is_available():
            check("CUDA version", True, torch.version.cuda)
            check("GPU", True, torch.cuda.get_device_name(0))
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            check("VRAM", vram >= 8, f"{vram:.1f} GB")
            
            # Quick allocation test
            try:
                x = torch.randn(1000, 1000, device='cuda')
                del x
                torch.cuda.empty_cache()
                check("CUDA allocation", True)
            except Exception as e:
                all_passed &= check("CUDA allocation", False, str(e))
    except ImportError:
        all_passed &= check("PyTorch installed", False)
    
    # Core packages
    print("\n[Core Packages]")
    packages = {
        'numpy': 'numpy',
        'opencv': 'cv2',
        'pyiqa': 'pyiqa',
        'lpips': 'lpips',
        'einops': 'einops',
        'timm': 'timm',
        'hydra': 'hydra',
        'wandb': 'wandb',
        'tqdm': 'tqdm',
    }
    
    for name, module in packages.items():
        try:
            m = __import__(module)
            version = getattr(m, '__version__', 'installed')
            check(name, True, version)
        except ImportError:
            all_passed &= check(name, False, "not installed")
    
    # FFmpeg
    print("\n[System Tools]")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        version = result.stdout.split('\n')[0].split('version')[1].split()[0] if 'version' in result.stdout else 'unknown'
        check("ffmpeg", True, version)
    except Exception:
        all_passed &= check("ffmpeg", False)
    
    # External models
    print("\n[External Models]")
    external_dir = Path("external")
    models = {
        'RIFE': 'Practical-RIFE',
        'VFIMamba': 'VFIMamba',
        'SAFA': 'WACV2024-SAFA',
        'SPAN': 'SPAN',
    }
    
    for name, dirname in models.items():
        path = external_dir / dirname
        check(name, path.exists(), str(path) if path.exists() else "not found")
    
    # RIFE weights
    print("\n[Model Weights]")
    rife_weights = external_dir / "Practical-RIFE" / "train_log"
    all_passed &= check("RIFE weights", rife_weights.exists())
    
    # VFIMamba weights (optional)
    vfimamba_weights = external_dir / "VFIMamba" / "checkpoints"
    check("VFIMamba weights", vfimamba_weights.exists(), "optional - manual download")
    
    # Project structure
    print("\n[Project Structure]")
    dirs = ['data/raw', 'data/processed', 'outputs/benchmarks', 'models', 'scripts']
    for d in dirs:
        check(d, Path(d).exists())
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print(" ✓ ALL CHECKS PASSED - Ready to go!")
    else:
        print(" ✗ SOME CHECKS FAILED - Review above")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
