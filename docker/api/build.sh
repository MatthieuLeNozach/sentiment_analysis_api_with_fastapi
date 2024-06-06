#!/bin/bash
SCRIPT_DIR=$(dirname "$0")

cd "$SCRIPT_DIR"

# Get the name of the containing folder
FOLDER_NAME=$(basename "$PWD")

# Get the Docker Hub account name from the command line argument
DOCKERHUB_ACCOUNT=$1
DOCKERHUB_REPO=$2

# Check if the Docker Hub account name is provided
if [ -z "$DOCKERHUB_ACCOUNT" ]; then
    echo "Please provide a Docker Hub account name as an argument."
    exit 1
fi

# Ensure the destination directory exists
mkdir -p ./api

# Copy the contents of the source api directory to the already created destination directory
cp -r ../../app/api/. ./api/
echo "Checking the '/api' content brought to image build context:"
ls ./api

# Ensure the common_utils directory exists
mkdir -p ./api/common_utils

# Copy the contents of the common_utils directory to the already created destination directory
cp -r ../../app/common_utils/. ./api/common_utils/
echo "Checking the '/api/common_utils' content brought to image build context:"
ls ./api/common_utils

# Build the Docker image with the folder name as the tag and the provided Docker Hub account name
docker build -t "${DOCKERHUB_ACCOUNT}/${DOCKERHUB_REPO}:${FOLDER_NAME}" -f "Dockerfile.${FOLDER_NAME}" .

# Cleanup: Remove copied directories
rm -rf ./api