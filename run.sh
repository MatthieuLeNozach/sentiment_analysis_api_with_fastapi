#!/bin/bash

set -e


THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

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

# install core and development Python dependencies into the currently activated venv
function install-requirements {
    python -m pip install --upgrade pip
    python -m pip install -r requirements_ci.txt
}

# run linting, formatting, and other static code quality tools
function lint {
    pre-commit run --all-files
}


function build:api() {
    source .env
    echo "running build-api..."
    bash docker/api/build.sh "${DOCKERHUB_ACCOUNT}" "${DOCKERHUB_REPO}"
}

function build:inference() {
    source .env
    echo "running build-inference..."
    bash docker/inference/build.sh "${DOCKERHUB_ACCOUNT}" "${DOCKERHUB_REPO}"
}


function build:all() {
    echo "Building all images..."
    build:api
    build:inference
}



# print all functions in this file
function help {
    echo "$0 <task> <args>"
    echo "Tasks:"
    compgen -A function | cat -n
}

TIMEFORMAT="Task completed in %3lR"
time ${@:-help}
