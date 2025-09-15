#!/usr/bin/env python3
"""
Install dependencies for the Fraud Detection System.
"""
import subprocess
import sys
import os
from pathlib import Path
import platform
import venv
import shutil

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 50)
    print(f" {text}".upper())
    print("=" * 50)

def run_command(command, cwd=None):
    """Run a shell command and print its output."""
    print(f"Running: {command}")
    try:
        # Use a list for the command to handle paths with spaces
        if isinstance(command, str):
            command = command.split()
            
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=str(cwd) if cwd else None,  # Convert Path to string
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False

def create_virtualenv(venv_path):
    """Create a Python virtual environment."""
    print(f"Creating virtual environment in {venv_path}...")
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return True
    
    try:
        venv.create(venv_path, with_pip=True)
        print("Virtual environment created successfully.")
        return True
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
        return False

def get_pip_command(venv_path):
    """Get the pip command for the virtual environment."""
    if platform.system() == "Windows":
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    return str(pip_path)

def install_python_packages(venv_path, requirements_files):
    """Install Python packages from requirements files."""
    pip_cmd = get_pip_command(venv_path)
    
    for req_file in requirements_files:
        if not req_file.exists():
            print(f"Requirements file not found: {req_file}")
            continue
            
        print(f"Installing packages from {req_file}...")
        # Use list form of command with proper path handling
        cmd = [pip_cmd, 'install', '-r', str(req_file)]
        if not run_command(cmd):
            print(f"Failed to install packages from {req_file}")
            return False
    
    # Install the package in development mode
    print("Installing package in development mode...")
    cmd = [pip_cmd, 'install', '-e', '.']
    if not run_command(cmd):
        print("Failed to install the package in development mode")
        return False
    
    return True

def create_directories():
    """Create necessary directories for the project."""
    print("Creating project directories...")
    dirs = [
        "data/raw",
        "data/processed",
        "data/models",
        "notebooks",
        "logs"
    ]
    
    for dir_path in dirs:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def main():
    """Main function to install dependencies and set up the project."""
    # Get project root
    project_root = Path(__file__).parent.absolute()
    venv_path = project_root / "venv"
    
    print_header("Setting up Fraud Detection System")
    
    # Create virtual environment
    if not create_virtualenv(venv_path):
        print("Failed to create virtual environment. Exiting.")
        sys.exit(1)
    
    # Create project directories
    create_directories()
    
    # Install Python packages
    requirements_files = [
        project_root / "requirements.txt",
        project_root / "dashboard" / "requirements-dashboard.txt"
    ]
    
    if not install_python_packages(venv_path, requirements_files):
        print("Failed to install some packages. Check the logs above for details.")
        sys.exit(1)
    
    print_header("Installation Complete!")
    print("\nTo activate the virtual environment, run:")
    if platform.system() == "Windows":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    
    print("\nTo run the dashboard:")
    print("  python run.py dashboard")
    
    print("\nTo run the complete pipeline (process data, train model, and launch dashboard):")
    print("  python run.py all")

if __name__ == "__main__":
    main()
