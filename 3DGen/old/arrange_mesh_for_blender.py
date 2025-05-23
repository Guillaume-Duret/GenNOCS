import os
import re
import argparse
from pathlib import Path

# Set up argument parser
parser = argparse.ArgumentParser(description="Organize files by base name into folders.")
parser.add_argument("input_dir", type=str, help="The input directory containing the files to organize.")
parser.add_argument("output_dir", type=str, help="The output directory where organized folders will be created.")

# Parse arguments
args = parser.parse_args()
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)

# Check if input directory exists
if not input_dir.exists() or not input_dir.is_dir():
    print(f"Error: The input directory '{input_dir}' does not exist or is not a directory.")
    exit(1)

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Regular expression to match the base name (e.g., "bowl2_0_1" from "bowl2_0_1.obj")
pattern = re.compile(r"^(bowl2_\d+_\d+)\..+$")

# Organize files into folders
for file in input_dir.iterdir():
    if file.is_file():
        match = pattern.match(file.name)
        if match:
            base_name = match.group(1)
            
            # Create a directory for the base name inside the output directory
            target_directory = output_dir / base_name
            target_directory.mkdir(exist_ok=True)
            
            # Move the file into the corresponding directory
            destination = target_directory / file.name
            file.rename(destination)

print("Files organized into folders by base name.")

