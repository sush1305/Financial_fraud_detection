#!/usr/bin/env python3
"""
Run the complete Fraud Detection System pipeline.

This script will:
1. Process the data
2. Train the model
3. Launch the dashboard
"""
import sys
import subprocess
import os
from pathlib import Path
import webbrowser
import time

def run_command(command, cwd=None):
    """Run a shell command and print its output."""
    print(f"\nRunning: {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
        # Wait for the process to complete
        process.wait()
        
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Run the complete pipeline."""
    # Get project root
    project_root = Path(__file__).parent.absolute()
    
    # Determine the Python interpreter to use
    python_cmd = sys.executable
    if not python_cmd:
        python_cmd = "python"
    
    print("=" * 50)
    print("FRAUD DETECTION SYSTEM - COMPLETE PIPELINE")
    print("=" * 50)
    
    # Step 1: Process data
    print("\n" + "=" * 50)
    print("STEP 1: PROCESSING DATA")
    print("=" * 50)
    if not run_command(f"{python_cmd} run.py process-data", cwd=project_root):
        print("\nError: Data processing failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Train model
    print("\n" + "=" * 50)
    print("STEP 2: TRAINING MODEL")
    print("=" * 50)
    if not run_command(f"{python_cmd} run.py train", cwd=project_root):
        print("\nError: Model training failed. Exiting.")
        sys.exit(1)
    
    # Step 3: Launch dashboard
    print("\n" + "=" * 50)
    print("STEP 3: LAUNCHING DASHBOARD")
    print("=" * 50)
    
    # Open the dashboard in the default web browser
    dashboard_url = "http://localhost:8050"
    print(f"\nThe dashboard will be available at: {dashboard_url}")
    print("Press Ctrl+C to stop the dashboard server.")
    
    # Start the dashboard in a new process
    dashboard_process = subprocess.Popen(
        [python_cmd, "run.py", "dashboard", "--no-browser"],
        cwd=project_root
    )
    
    # Give the server a moment to start
    time.sleep(2)
    
    # Open the dashboard in the default web browser
    webbrowser.open(dashboard_url)
    
    try:
        # Keep the script running until interrupted
        dashboard_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
        dashboard_process.terminate()
        print("Done!")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
