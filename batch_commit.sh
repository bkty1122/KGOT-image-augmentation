#!/bin/bash

# This script adds, commits, and pushes files in batches of 1000 until all changes are processed.

# Configuration
BATCH_SIZE=1000   # Number of files to process per batch
BRANCH_NAME="main"  # Change this if your branch is not "main"
COMMIT_PREFIX="Batch commit"  # Prefix for commit messages

# Get the total number of files to process
function get_file_list() {
    git status --porcelain | awk '{print $2}'
}

# Add the first N files from the file list
function stage_files() {
    local files=("$@")  # List of files to stage
    for file in "${files[@]}"; do
        git add "$file"
    done
}

# Commit and push changes
function commit_and_push() {
    local batch_number=$1

    # Commit the changes
    git commit -m "${COMMIT_PREFIX} #${batch_number}"

    # Push the changes to the remote branch
    git push origin "$BRANCH_NAME"
}

# Main loop
function process_batches() {
    local batch_number=1
    while true; do
        echo "Processing batch #$batch_number..."

        # Get a list of unstaged files
        file_list=($(get_file_list))

        # If no files are left, exit the loop
        if [ ${#file_list[@]} -eq 0 ]; then
            echo "All changes have been processed!"
            break
        fi

        # Get the first $BATCH_SIZE files from the list
        files_to_stage=("${file_list[@]:0:$BATCH_SIZE}")

        # Stage the files
        echo "Staging ${#files_to_stage[@]} files..."
        stage_files "${files_to_stage[@]}"

        # Commit and push the changes
        echo "Committing and pushing batch #$batch_number..."
        commit_and_push "$batch_number"

        # Increment the batch number
        ((batch_number++))
    done
}

# Start processing files in batches
process_batches