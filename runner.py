#!/usr/bin/env python3
import subprocess
import sys
import os
import time

def run_script(script_name):
    """Run a Python script and handle any errors."""
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}")
    
    try:
        # Use the same Python interpreter that's running this script
        python_exe = sys.executable
        result = subprocess.run([python_exe, script_name], check=True)
        
        if result.returncode == 0:
            print(f"\n✅ {script_name} executed successfully!")
        else:
            print(f"\n❌ {script_name} failed with return code {result.returncode}")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Error: {script_name} not found in the current directory")
        return False
    
    return True

def main():
    """Run track.py, plots.py, and guess.py in sequence."""
    # Check if scripts exist
    scripts = ["track.py", "plots.py", "guess.py"]
    for script in scripts:
        if not os.path.exists(script):
            print(f"Error: {script} not found in the current directory")
            sys.exit(1)
    
    print("Starting sequential execution of scripts...")
    
    # Run scripts in sequence
    if run_script("track.py"):
        time.sleep(1)  # Brief pause between scripts
        
        if run_script("plots.py"):
            time.sleep(1)  # Brief pause between scripts
            
            if run_script("guess.py"):
                print("\n🎉 All scripts executed successfully!")
            else:
                print("\n❌ Script sequence stopped at guess.py")
        else:
            print("\n❌ Script sequence stopped at plots.py")
    else:
        print("\n❌ Script sequence stopped at track.py")

if __name__ == "__main__":
    main()
