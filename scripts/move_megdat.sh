#!/bin/bash

# This needs to be ran from my local machine, as that's where the files are.
# Set source directory (current directory) and target directory
SRC_DIR="."
TARGET_DIR="/Volumes/project/3018078.02/MEG_ingmar"

# Loop over all matching .mat files
for file in ${SRC_DIR}/sub*_100Hz_badmuscle_badlowfreq_badcomp.mat; do
    if [[ -f "$file" ]]; then
        base=$(basename "$file")
        if [[ -f "$TARGET_DIR/$base" ]]; then
            echo "Skipping $base (already exists in target)"
        else
            echo "Copying $file to $TARGET_DIR"
            cp -pR "$file" "$TARGET_DIR"
        fi
    fi
done

echo "Copying complete."