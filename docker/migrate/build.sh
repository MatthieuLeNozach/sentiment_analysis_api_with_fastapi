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
mkdir -p ./alembic

# Copy the contents of the alembic directory to the already created destination directory
cp -r ../../alembic/. ./alembic/
echo "Checking the '/alembic' content brought to image build context:"
ls 

# Copy the alembic.ini file to the destination directory
cp ../../alembic.ini ./alembic.ini
echo "Checking the '/alembic.ini' content brought to image build context:"
ls 


touch ./__init__.py

# Build the Docker image with the folder name as the tag and the provided Docker Hub account name
docker build -t "${DOCKERHUB_ACCOUNT}/${DOCKERHUB_REPO}:${FOLDER_NAME}" -f "Dockerfile.${FOLDER_NAME}" --build-arg BASE_IMAGE="${DOCKERHUB_ACCOUNT}/${DOCKERHUB_REPO}:api" .

# Cleanup: Remove copied directories
#rm -rf ./alembic
#rm -f ./alembic.ini
#rm -rf ./app