#!/bin/bash

set -e

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#================================================================#
#####################    ENV VARIABLES    ########################

source .env/.dev-sample

# export the contents of .env as environment variables
function try-load-dotenv {
    if [ ! -f "$THIS_DIR/.env" ]; then
        echo "no .env file found"
        return 1
    fi

    while read -r line; do
        export "$line"
    done < <(grep -v '^#' "$THIS_DIR/.env" | grep -v '^$')
}

#================================================================#
########################    DOCKER    ############################

# Function to build a single Docker image
build_image() {
    local target_dir="$1"

    # Load environment variables from .env file
    source "$THIS_DIR/.env/.dev-sample" || { echo "Failed to load environment variables from .env/.dev-sample"; return 1; }

    # Check if DOCKERHUB_ACCOUNT and DOCKERHUB_REPO are set
    if [ -z "$DOCKERHUB_ACCOUNT" ] || [ -z "$DOCKERHUB_REPO" ]; then
        echo "DOCKERHUB_ACCOUNT and DOCKERHUB_REPO must be set in .env/.dev-sample"
        return 1
    fi

    # Get the name of the containing folder
    folder_name=$(basename "$target_dir")

    # Construct the Dockerfile path
    dockerfile_path="${target_dir}/Dockerfile.${folder_name}"

    # Build the Docker image
    if [ "$folder_name" = "base" ]; then
        docker build -t "${DOCKERHUB_ACCOUNT}/${DOCKERHUB_REPO}:${folder_name}" -f "$dockerfile_path" .
    else
        docker build --build-arg BASE_IMAGE="${DOCKERHUB_ACCOUNT}/${DOCKERHUB_REPO}:base" -t "${DOCKERHUB_ACCOUNT}/${DOCKERHUB_REPO}:${folder_name}" -f "$dockerfile_path" .
    fi
}
# Function to build all Docker images
build_all() {
    # Define the target directories and whether they should be built from the base image
    declare -A target_dirs=(
        ["compose/fastapi-celery/base"]=false
        ["compose/fastapi-celery/web"]=true
        ["compose/fastapi-celery/celery/worker"]=true
        ["compose/fastapi-celery/celery/beat"]=true
        ["compose/fastapi-celery/celery/flower"]=true
    )

    # Loop through the target directories and build the images
    for target_dir in "${!target_dirs[@]}"; do
        from_base="${target_dirs[$target_dir]}"
        build_image "$THIS_DIR/$target_dir" "$from_base"
    done
}

# Function to push a single Docker image
push_image() {
    local service_dir=$1
    local dockerhub_account=$2
    local dockerhub_repo=$3

    # Get the name of the containing folder
    folder_name=$(basename "$service_dir")

    # Push the Docker image
    docker push "${dockerhub_account}/${dockerhub_repo}:${folder_name}"
}


# Function to push all Docker images
push_all_images() {
    local dockerhub_account=$1
    local dockerhub_repo=$2

    # Define the services and their corresponding Dockerfiles
    declare -A services
    services=(
        ["compose/fastapi-celery/web"]="Dockerfile"
        ["compose/fastapi-celery/celery/worker"]="Dockerfile.worker"
        ["compose/fastapi-celery/celery/beat"]="Dockerfile.beat"
        ["compose/fastapi-celery/celery/flower"]="Dockerfile.flower"
    )

    # Loop through the services and push the images
    for service_dir in "${!services[@]}"; do
        push_image "$service_dir" "$dockerhub_account" "$dockerhub_repo"
    done
}


#================================================================#
########################    LINTING    ###########################


# run linting, formatting, and other static code quality tools
function lint {
    pre-commit run --all-files
}



#================================================================#
########################    TESTS    #############################




#================================================================#
########################    UTILS    #############################

function purge:pycache() {
    find . -type d -name "__pycache__" -exec sudo rm -r {} +
    find . -type f -name "*.pyc" -exec sudo rm -f {} +
}


# print all functions in this file
function help {
    echo "$0 <task> <args>"
    echo "Tasks:"
    compgen -A function | cat -n
}



TIMEFORMAT="Task completed in %3lR"
time ${@:-help}