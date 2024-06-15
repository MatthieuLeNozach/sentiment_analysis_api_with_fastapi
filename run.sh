#!/bin/bash

set -e

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#================================================================#
#####################    ENV VARIABLES    ########################

# Function to load environment variables from [.env/.dev-sample]
function try-load-dotenv {
    if [ ! -f "$THIS_DIR/.env/.dev-sample" ]; then
        echo "no .env/.dev-sample file found"
        return 1
    fi

    while read -r line; do
        export "$line"
    done < <(grep -v '^#' "$THIS_DIR/.env/.dev-sample" | grep -v '^$')
}

#================================================================#
########################    DOCKER    ############################

# Function to build a single Docker image
function build-image() {
    local target_dir="$1"

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
function build-all() {
    # Load environment variables
    try-load-dotenv || { echo "Failed to load environment variables"; return 1; }

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
        build-image "$THIS_DIR/$target_dir" "$from_base"
    done
}

# Function to push a single Docker image
function push-image() {
    local service_dir="$1"

    # Load environment variables
    try-load-dotenv || { echo "Failed to load environment variables"; return 1; }

    # Get the name of the containing folder
    folder_name=$(basename "$service_dir")

    # Push the Docker image
    docker push "${DOCKERHUB_ACCOUNT}/${DOCKERHUB_REPO}:${folder_name}"
}

# Function to push all Docker images
function push-all-images() {
    # Load environment variables
    try-load-dotenv || { echo "Failed to load environment variables"; return 1; }

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
        push-image "$THIS_DIR/$service_dir"
    done
}


#================================================================#
#######################    DATABASE    ###########################

function init-alembic() {
    alembic init -t async alembic
}

function run-revision-postgres() {
    try-load-dotenv || { echo "Failed to load environment variables"; return 1; }

    # Start PostgreSQL service
    docker compose up -d postgres

    # Wait for PostgreSQL to be ready
    until docker compose exec postgres pg_isready -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -p "$POSTGRES_PORT"; do
        >&2 echo "Waiting for PostgreSQL to become available..."
        sleep 1
    done

    # Run Alembic revision in the web container
    docker compose run web alembic revision --autogenerate 

    # Stop all services
    docker compose down
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