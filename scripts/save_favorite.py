import shutil
import sys
import os
from datetime import datetime
from pathlib import Path

def save_favorite(description):
    # Source file is always the latest output_test.mp4
    source = Path("output_test.mp4")
    
    if not source.exists():
        print(f"Error: {source} not found. Run the pipeline first.")
        sys.exit(1)
        
    # Create favorites dir if not exists
    favorites_dir = Path("favorites")
    favorites_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Clean description (replace spaces with underscores, remove weird chars)
    clean_desc = "".join(c if c.isalnum() or c in "-_" else "_" for c in description)
    clean_desc = clean_desc.strip("_")
    
    # Form new filename
    filename = f"{timestamp}_{clean_desc}.mp4"
    dest = favorites_dir / filename
    
    # Copy
    shutil.copy2(source, dest)
    print(f"✅ Saved favorite to: {dest}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/save_favorite.py \"description of run\"")
        sys.exit(1)
        
    description = sys.argv[1]
    save_favorite(description)
