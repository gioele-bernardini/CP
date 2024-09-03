#!/usr/bin/env bash

for folder in */; do
  if [ -d "$folder" ]; then
    # Remove the trailing slash from the folder name
    folder=${folder%/}

    # Create subdirectories if they do not already exist
    for difficulty in "easy" "medium" "hard"; do
      mkdir -p "$folder/$difficulty"
      touch "$folder/$difficulty/.gitkeep"
    done
  fi
done

echo "Folder structure correctly set up"


