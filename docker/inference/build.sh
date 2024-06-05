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
mkdir -p ./inference

# Copy the contents of the source inference directory to the already created destination directory
cp -r ../../app/inference/. ./inference/

# Ensure the common_utils directory exists
mkdir -p ./inference/common_utils

# Copy the contents of the common_utils directory to the already created destination directory
cp -r ../../app/common_utils/. ./inference/common_utils/
echo "Checking the '/inference/common_utils' content brought to image build context:"
ls ./inference/common_utils

# Copy model weights to the build context
if [ -d "../../models/" ]; then
    echo "Models directory found. Copying the model weights..."
    ls ../../models/
    mkdir -p ./models
    cp -r ../../models/ ./models/
else
    echo "Models directory not found. Skipping the copy operation..."
fi

# Build the Docker image with the folder name as the tag and the provided Docker Hub account name
docker build -t "${DOCKERHUB_ACCOUNT}/${DOCKERHUB_REPO}:${FOLDER_NAME}" -f "Dockerfile.${FOLDER_NAME}" --build-arg BASE_IMAGE="${DOCKERHUB_ACCOUNT}/bird-sound-classif:base" .

# Cleanup: Remove copied directories
rm -rf ./inference
rm -rf ./models