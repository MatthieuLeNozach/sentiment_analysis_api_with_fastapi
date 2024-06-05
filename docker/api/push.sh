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

docker login

docker push "${DOCKERHUB_ACCOUNT}/${DOCKERHUB_REPO}:${FOLDER_NAME}"
