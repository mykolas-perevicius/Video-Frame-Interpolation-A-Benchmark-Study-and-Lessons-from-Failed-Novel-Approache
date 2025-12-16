#!/usr/bin/env python3
"""
probe_system.py - System Compatibility Probe for Gaming VFI+SR Project

Run this FIRST before any setup to understand your environment.

Usage:
    python probe_system.py
    python probe_system.py --json  # Output as JSON for automation
    python probe_system.py --verbose  # Extra details

Works on: Windows, WSL2, Native Linux
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


# ==============================================================================
# Data Classes for Results
# ==============================================================================

@dataclass
class PlatformInfo:
    os_type: str = ""  # "Windows", "Linux", "WSL2", "Darwin"
    os_version: str = ""
    hostname: str = ""
    architecture: str = ""
    is_wsl: bool = False
    wsl_version: Optional[int] = None
    windows_build: Optional[str] = None
    kernel_version: Optional[str] = None


@dataclass
class GPUInfo:
    available: bool = False
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    vram_total_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None
    compute_capability: Optional[str] = None
    nvidia_smi_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PythonInfo:
    version: str = ""
    executable: str = ""
    is_venv: bool = False
    venv_path: Optional[str] = None
    pip_version: Optional[str] = None
    packages_installed: Dict[str, str] = field(default_factory=dict)


@dataclass
class DependencyInfo:
    name: str
    required: bool
    found: bool
    version: Optional[str] = None
    path: Optional[str] = None
    notes: Optional[str] = None


@dataclass 
class DiskInfo:
    path: str
    total_gb: float
    free_gb: float
    is_wsl_mount: bool = False  # True if /mnt/c, /mnt/d, etc.


@dataclass
class MemoryInfo:
    total_gb: float
    available_gb: float
    swap_total_gb: float = 0
    swap_free_gb: float = 0


@dataclass
class ProbeResult:
    timestamp: str
    platform: PlatformInfo
    gpu: GPUInfo
    python: PythonInfo
    memory: MemoryInfo
    disks: List[DiskInfo]
    dependencies: List[DependencyInfo]
    recommendations: List[str]
    warnings: List[str]
    errors: List[str]
    compatibility_score: int  # 0-100
    recommended_setup: str  # "wsl2", "native_linux", "native_windows", "not_compatible"


# ==============================================================================
# Probe Functions
# ==============================================================================

def probe_platform() -> PlatformInfo:
    """Detect platform and environment details"""
    info = PlatformInfo()
    
    info.hostname = platform.node()
    info.architecture = platform.machine()
    
    system = platform.system()
    
    if system == "Linux":
        # Check if WSL
        try:
            with open("/proc/version", "r") as f:
                version_info = f.read().lower()
                if "microsoft" in version_info or "wsl" in version_info:
                    info.is_wsl = True
                    info.os_type = "WSL2"
                    
                    # Detect WSL version
                    if "wsl2" in version_info:
                        info.wsl_version = 2
                    else:
                        # Check for WSL2 indicators
                        try:
                            result = subprocess.run(
                                ["cat", "/proc/sys/fs/binfmt_misc/WSLInterop"],
                                capture_output=True, text=True
                            )
                            info.wsl_version = 2 if result.returncode == 0 else 1
                        except:
                            info.wsl_version = 2  # Assume WSL2 if uncertain
                else:
                    info.os_type = "Linux"
        except:
            info.os_type = "Linux"
        
        # Get kernel version
        info.kernel_version = platform.release()
        
        # Get distro info
        try:
            result = subprocess.run(
                ["lsb_release", "-d", "-s"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                info.os_version = result.stdout.strip()
            else:
                info.os_version = platform.release()
        except:
            info.os_version = platform.release()
            
    elif system == "Windows":
        info.os_type = "Windows"
        info.os_version = platform.version()
        info.windows_build = platform.win32_ver()[1]
        
    elif system == "Darwin":
        info.os_type = "macOS"
        info.os_version = platform.mac_ver()[0]
    
    return info


def probe_gpu() -> GPUInfo:
    """Detect NVIDIA GPU and CUDA availability"""
    info = GPUInfo()
    
    # Find nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        # Try common paths
        common_paths = [
            "/usr/bin/nvidia-smi",
            "/usr/local/bin/nvidia-smi",
            "C:\\Windows\\System32\\nvidia-smi.exe",
            "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                nvidia_smi = path
                break
    
    if not nvidia_smi:
        info.error = "nvidia-smi not found - NVIDIA driver may not be installed"
        return info
    
    info.nvidia_smi_path = nvidia_smi
    
    try:
        # Query GPU info
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            info.error = f"nvidia-smi failed: {result.stderr}"
            return info
        
        # Parse output (may have multiple GPUs)
        lines = result.stdout.strip().split("\n")
        info.gpu_count = len(lines)
        
        if lines:
            # Use first GPU
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 5:
                info.gpu_name = parts[0]
                info.driver_version = parts[1]
                info.vram_total_mb = int(float(parts[2]))
                info.vram_free_mb = int(float(parts[3]))
                info.compute_capability = parts[4]
        
        info.available = True
        
        # Try to get CUDA version
        result = subprocess.run(
            [nvidia_smi],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "CUDA Version:" in result.stdout:
            for line in result.stdout.split("\n"):
                if "CUDA Version:" in line:
                    info.cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                    break
                    
    except subprocess.TimeoutExpired:
        info.error = "nvidia-smi timed out"
    except Exception as e:
        info.error = str(e)
    
    return info


def probe_python() -> PythonInfo:
    """Get Python environment details"""
    info = PythonInfo()
    
    info.version = platform.python_version()
    info.executable = sys.executable
    
    # Check if in venv
    info.is_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    
    if info.is_venv:
        info.venv_path = sys.prefix
    
    # Get pip version
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            info.pip_version = result.stdout.split()[1]
    except:
        pass
    
    # Check key packages
    key_packages = [
        "torch", "torchvision", "numpy", "opencv-python", 
        "ffmpeg-python", "pyiqa", "lpips"
    ]
    
    for pkg in key_packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Version:"):
                        info.packages_installed[pkg] = line.split(":")[1].strip()
                        break
        except:
            pass
    
    return info


def probe_memory() -> MemoryInfo:
    """Get system memory information"""
    info = MemoryInfo(total_gb=0, available_gb=0)
    
    system = platform.system()
    
    if system in ["Linux", "Darwin"]:
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
                
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    info.total_gb = int(line.split()[1]) / 1024 / 1024
                elif line.startswith("MemAvailable:"):
                    info.available_gb = int(line.split()[1]) / 1024 / 1024
                elif line.startswith("SwapTotal:"):
                    info.swap_total_gb = int(line.split()[1]) / 1024 / 1024
                elif line.startswith("SwapFree:"):
                    info.swap_free_gb = int(line.split()[1]) / 1024 / 1024
        except:
            pass
            
    elif system == "Windows":
        try:
            import ctypes
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            
            info.total_gb = stat.ullTotalPhys / 1024 / 1024 / 1024
            info.available_gb = stat.ullAvailPhys / 1024 / 1024 / 1024
        except:
            pass
    
    return info


def probe_disks() -> List[DiskInfo]:
    """Get disk space information for relevant paths"""
    disks = []
    
    # Paths to check
    paths_to_check = [
        Path.home(),
        Path.cwd(),
    ]
    
    # On WSL, also check Windows mounts
    if platform.system() == "Linux":
        for mount in ["/mnt/c", "/mnt/d", "/mnt/e"]:
            if os.path.exists(mount):
                paths_to_check.append(Path(mount))
    
    checked_devices = set()
    
    for path in paths_to_check:
        try:
            stat = os.statvfs(path)
            
            # Avoid duplicate entries for same filesystem
            device_id = (stat.f_fsid if hasattr(stat, 'f_fsid') else str(path))
            if device_id in checked_devices:
                continue
            checked_devices.add(device_id)
            
            total_gb = (stat.f_blocks * stat.f_frsize) / 1024 / 1024 / 1024
            free_gb = (stat.f_bavail * stat.f_frsize) / 1024 / 1024 / 1024
            
            is_wsl_mount = str(path).startswith("/mnt/") and len(str(path)) == 6
            
            disks.append(DiskInfo(
                path=str(path),
                total_gb=round(total_gb, 1),
                free_gb=round(free_gb, 1),
                is_wsl_mount=is_wsl_mount,
            ))
        except:
            pass
    
    return disks


def probe_dependencies() -> List[DependencyInfo]:
    """Check for required system dependencies"""
    deps = []
    
    # FFmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_version = None
    if ffmpeg_path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True
            )
            ffmpeg_version = result.stdout.split("\n")[0].split("version")[1].split()[0]
        except:
            pass
    
    deps.append(DependencyInfo(
        name="ffmpeg",
        required=True,
        found=ffmpeg_path is not None,
        version=ffmpeg_version,
        path=ffmpeg_path,
    ))
    
    # Git
    git_path = shutil.which("git")
    git_version = None
    if git_path:
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            git_version = result.stdout.strip().split()[-1]
        except:
            pass
    
    deps.append(DependencyInfo(
        name="git",
        required=True,
        found=git_path is not None,
        version=git_version,
        path=git_path,
    ))
    
    # wget or curl
    wget_path = shutil.which("wget")
    curl_path = shutil.which("curl")
    
    deps.append(DependencyInfo(
        name="wget",
        required=False,
        found=wget_path is not None,
        path=wget_path,
        notes="or curl" if not wget_path and curl_path else None,
    ))
    
    # unzip
    unzip_path = shutil.which("unzip")
    deps.append(DependencyInfo(
        name="unzip",
        required=True,
        found=unzip_path is not None,
        path=unzip_path,
    ))
    
    # Check Python 3.10+
    python_ok = sys.version_info >= (3, 10)
    deps.append(DependencyInfo(
        name="Python 3.10+",
        required=True,
        found=python_ok,
        version=platform.python_version(),
        path=sys.executable,
        notes=None if python_ok else "Python 3.10 or higher required",
    ))
    
    return deps


def check_pytorch_cuda() -> Dict[str, Any]:
    """Deep check of PyTorch CUDA functionality"""
    result = {
        "pytorch_installed": False,
        "pytorch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "device_name": None,
        "can_allocate": False,
        "error": None,
    }
    
    try:
        import torch
        result["pytorch_installed"] = True
        result["pytorch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            result["cuda_version"] = torch.version.cuda
            result["cudnn_version"] = str(torch.backends.cudnn.version())
            result["device_name"] = torch.cuda.get_device_name(0)
            
            # Try to allocate tensor
            try:
                x = torch.randn(1000, 1000, device='cuda')
                result["can_allocate"] = True
                del x
                torch.cuda.empty_cache()
            except Exception as e:
                result["error"] = f"Allocation failed: {e}"
                
    except ImportError:
        result["error"] = "PyTorch not installed"
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ==============================================================================
# Analysis and Recommendations
# ==============================================================================

def analyze_results(
    platform_info: PlatformInfo,
    gpu_info: GPUInfo,
    python_info: PythonInfo,
    memory_info: MemoryInfo,
    disks: List[DiskInfo],
    dependencies: List[DependencyInfo],
) -> tuple:
    """Analyze probe results and generate recommendations"""
    
    recommendations = []
    warnings = []
    errors = []
    score = 100
    recommended_setup = "unknown"
    
    # Platform analysis
    if platform_info.os_type == "WSL2":
        if platform_info.wsl_version == 1:
            errors.append("WSL1 detected - need WSL2 for GPU support")
            score -= 50
        else:
            recommendations.append("WSL2 detected - good choice for this project")
            recommended_setup = "wsl2"
            
    elif platform_info.os_type == "Linux":
        recommendations.append("Native Linux detected - optimal for ML workloads")
        recommended_setup = "native_linux"
        
    elif platform_info.os_type == "Windows":
        warnings.append("Native Windows detected - WSL2 recommended for easier setup")
        recommendations.append("Consider using WSL2: wsl --install -d Ubuntu-22.04")
        recommended_setup = "native_windows"
        score -= 10
        
    elif platform_info.os_type == "macOS":
        errors.append("macOS detected - NVIDIA GPU not supported")
        score -= 80
        recommended_setup = "not_compatible"
    
    # GPU analysis
    if not gpu_info.available:
        errors.append(f"No NVIDIA GPU detected: {gpu_info.error}")
        score -= 50
    else:
        if gpu_info.vram_total_mb and gpu_info.vram_total_mb < 8000:
            warnings.append(f"GPU VRAM ({gpu_info.vram_total_mb}MB) is low - may need to reduce batch sizes")
            score -= 10
        elif gpu_info.vram_total_mb and gpu_info.vram_total_mb >= 20000:
            recommendations.append(f"Excellent GPU VRAM ({gpu_info.vram_total_mb}MB) - can run all models at full capacity")
        
        if gpu_info.driver_version:
            try:
                major_version = int(gpu_info.driver_version.split(".")[0])
                if major_version < 470:
                    errors.append(f"NVIDIA driver {gpu_info.driver_version} too old - need 470.76+")
                    score -= 30
                elif major_version < 530:
                    warnings.append(f"NVIDIA driver {gpu_info.driver_version} is older - consider updating to 535+")
            except:
                pass
        
        if gpu_info.cuda_version:
            try:
                cuda_major = int(gpu_info.cuda_version.split(".")[0])
                if cuda_major < 11:
                    warnings.append(f"CUDA {gpu_info.cuda_version} is old - recommend CUDA 12.1+")
                    score -= 10
            except:
                pass
    
    # Memory analysis
    if memory_info.total_gb < 16:
        warnings.append(f"System RAM ({memory_info.total_gb:.1f}GB) is low - 32GB recommended")
        score -= 10
    elif memory_info.total_gb >= 32:
        recommendations.append(f"Good system RAM ({memory_info.total_gb:.1f}GB)")
    
    if memory_info.available_gb < 8:
        warnings.append(f"Available RAM ({memory_info.available_gb:.1f}GB) is low - close other applications")
    
    # Disk analysis
    best_disk = None
    for disk in disks:
        if disk.is_wsl_mount:
            warnings.append(f"WSL mount {disk.path} detected - avoid for processing (use ~/)")
        elif disk.free_gb < 50:
            warnings.append(f"Low disk space on {disk.path}: {disk.free_gb:.1f}GB free")
            score -= 5
        else:
            if best_disk is None or disk.free_gb > best_disk.free_gb:
                best_disk = disk
    
    if best_disk:
        if best_disk.free_gb >= 100:
            recommendations.append(f"Good disk space on {best_disk.path}: {best_disk.free_gb:.1f}GB free")
        else:
            warnings.append(f"Consider freeing disk space - have {best_disk.free_gb:.1f}GB, recommend 100GB+")
    
    # Dependency analysis
    missing_required = [d for d in dependencies if d.required and not d.found]
    if missing_required:
        for dep in missing_required:
            errors.append(f"Missing required dependency: {dep.name}")
            score -= 15
    
    # Python version
    if sys.version_info < (3, 10):
        errors.append(f"Python {platform.python_version()} too old - need 3.10+")
        score -= 20
    
    # Final score clamping
    score = max(0, min(100, score))
    
    # Determine if compatible
    if score < 30:
        recommended_setup = "not_compatible"
    
    return recommendations, warnings, errors, score, recommended_setup


# ==============================================================================
# Output Formatting
# ==============================================================================

def print_section(title: str, char: str = "="):
    """Print a section header"""
    width = 60
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_probe_results(result: ProbeResult, verbose: bool = False):
    """Print human-readable probe results"""
    
    print_section("SYSTEM COMPATIBILITY PROBE", "=")
    print(f"Timestamp: {result.timestamp}")
    
    # Platform
    print_section("PLATFORM", "-")
    p = result.platform
    print(f"  OS Type:      {p.os_type}")
    print(f"  OS Version:   {p.os_version}")
    print(f"  Architecture: {p.architecture}")
    if p.is_wsl:
        print(f"  WSL Version:  {p.wsl_version}")
    if p.kernel_version:
        print(f"  Kernel:       {p.kernel_version}")
    
    # GPU
    print_section("GPU", "-")
    g = result.gpu
    if g.available:
        print(f"  ✓ GPU Found:    {g.gpu_name}")
        print(f"    Driver:       {g.driver_version}")
        print(f"    CUDA:         {g.cuda_version}")
        print(f"    VRAM Total:   {g.vram_total_mb} MB")
        print(f"    VRAM Free:    {g.vram_free_mb} MB")
        print(f"    Compute Cap:  {g.compute_capability}")
        if g.gpu_count > 1:
            print(f"    GPU Count:    {g.gpu_count}")
    else:
        print(f"  ✗ GPU NOT AVAILABLE")
        print(f"    Error: {g.error}")
    
    # Memory
    print_section("MEMORY", "-")
    m = result.memory
    print(f"  RAM Total:     {m.total_gb:.1f} GB")
    print(f"  RAM Available: {m.available_gb:.1f} GB")
    if m.swap_total_gb > 0:
        print(f"  Swap Total:    {m.swap_total_gb:.1f} GB")
    
    # Disks
    print_section("DISK SPACE", "-")
    for d in result.disks:
        marker = "⚠" if d.is_wsl_mount else "✓"
        note = " (WSL mount - slow)" if d.is_wsl_mount else ""
        print(f"  {marker} {d.path}: {d.free_gb:.1f} GB free / {d.total_gb:.1f} GB total{note}")
    
    # Python
    print_section("PYTHON", "-")
    py = result.python
    status = "✓" if sys.version_info >= (3, 10) else "✗"
    print(f"  {status} Version:    {py.version}")
    print(f"    Executable: {py.executable}")
    print(f"    In venv:    {py.is_venv}")
    if py.packages_installed:
        print(f"    Key packages installed:")
        for pkg, ver in py.packages_installed.items():
            print(f"      - {pkg}: {ver}")
    
    # Dependencies
    print_section("DEPENDENCIES", "-")
    for dep in result.dependencies:
        status = "✓" if dep.found else "✗"
        ver = f" ({dep.version})" if dep.version else ""
        note = f" - {dep.notes}" if dep.notes else ""
        req = " [REQUIRED]" if dep.required and not dep.found else ""
        print(f"  {status} {dep.name}{ver}{note}{req}")
    
    # Recommendations
    if result.recommendations:
        print_section("RECOMMENDATIONS", "-")
        for rec in result.recommendations:
            print(f"  → {rec}")
    
    # Warnings
    if result.warnings:
        print_section("WARNINGS", "-")
        for warn in result.warnings:
            print(f"  ⚠ {warn}")
    
    # Errors
    if result.errors:
        print_section("ERRORS", "-")
        for err in result.errors:
            print(f"  ✗ {err}")
    
    # Summary
    print_section("SUMMARY", "=")
    
    score = result.compatibility_score
    if score >= 80:
        score_color = "EXCELLENT"
    elif score >= 60:
        score_color = "GOOD"
    elif score >= 40:
        score_color = "FAIR"
    else:
        score_color = "POOR"
    
    print(f"  Compatibility Score: {score}/100 ({score_color})")
    print(f"  Recommended Setup:   {result.recommended_setup.upper()}")
    
    # Next steps
    print_section("NEXT STEPS", "-")
    
    if result.recommended_setup == "wsl2":
        print("  1. Ensure you're working in WSL filesystem (~/), not /mnt/c/")
        print("  2. Run the WSL2 setup script:")
        print("     ./setup_wsl.sh")
        print("  3. Copy videos from Windows:")
        print("     cp /mnt/c/Users/YOU/Videos/clip.mp4 ~/gaming-vfisr/data/raw/")
        
    elif result.recommended_setup == "native_linux":
        print("  1. Run the Linux setup script:")
        print("     ./setup_linux.sh")
        print("  2. Place videos in data/raw/")
        
    elif result.recommended_setup == "native_windows":
        print("  1. RECOMMENDED: Install WSL2 instead:")
        print("     wsl --install -d Ubuntu-22.04")
        print("  2. OR run Windows setup (more complex):")
        print("     python setup_windows.py")
        
    elif result.recommended_setup == "not_compatible":
        print("  ✗ System does not meet minimum requirements")
        print("  Please address the errors above before proceeding")
    
    print("")


# ==============================================================================
# Main
# ==============================================================================

def run_probe(verbose: bool = False) -> ProbeResult:
    """Run all probes and return results"""
    from datetime import datetime
    
    print("Probing system... ", end="", flush=True)
    
    platform_info = probe_platform()
    print("platform... ", end="", flush=True)
    
    gpu_info = probe_gpu()
    print("GPU... ", end="", flush=True)
    
    python_info = probe_python()
    print("Python... ", end="", flush=True)
    
    memory_info = probe_memory()
    print("memory... ", end="", flush=True)
    
    disks = probe_disks()
    print("disks... ", end="", flush=True)
    
    dependencies = probe_dependencies()
    print("dependencies... ", end="", flush=True)
    
    # Analyze
    recommendations, warnings, errors, score, recommended_setup = analyze_results(
        platform_info, gpu_info, python_info, memory_info, disks, dependencies
    )
    print("done!")
    
    return ProbeResult(
        timestamp=datetime.now().isoformat(),
        platform=platform_info,
        gpu=gpu_info,
        python=python_info,
        memory=memory_info,
        disks=disks,
        dependencies=dependencies,
        recommendations=recommendations,
        warnings=warnings,
        errors=errors,
        compatibility_score=score,
        recommended_setup=recommended_setup,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Probe system compatibility for Gaming VFI+SR project"
    )
    parser.add_argument(
        "--json", 
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Show verbose output"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save results to file"
    )
    
    args = parser.parse_args()
    
    # Run probe
    result = run_probe(verbose=args.verbose)
    
    # Output
    if args.json:
        # Convert to JSON-serializable dict
        result_dict = asdict(result)
        output = json.dumps(result_dict, indent=2)
        print(output)
    else:
        print_probe_results(result, verbose=args.verbose)
    
    # Save to file if requested
    if args.output:
        result_dict = asdict(result)
        with open(args.output, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit code based on compatibility
    if result.compatibility_score < 30:
        sys.exit(2)  # Not compatible
    elif result.errors:
        sys.exit(1)  # Has errors but might work
    else:
        sys.exit(0)  # Good to go


if __name__ == "__main__":
    main()