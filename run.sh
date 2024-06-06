#!/bin/bash

set -e
source .env

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

# Transform init.sql.template into init.sql
function transform:init-sql() {
    try-load-dotenv

    if [ ! -f "$THIS_DIR/docker/postgres/init.sql.template" ]; then
        echo "init.sql.template file not found"
        return 1
    fi

    envsubst < "$THIS_DIR/docker/postgres/init.sql.template" > "$THIS_DIR/docker/postgres/init.sql"
    echo "init.sql file created successfully"
}

# install core and development Python dependencies into the currently activated venv
function install-requirements() {
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

function build:migrate() {
    source .env
    echo "running build-migrate..."
    bash docker/migrate/build.sh "${DOCKERHUB_ACCOUNT}" "${DOCKERHUB_REPO}"
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
    build:migrate
}

function push:api() {
    source .env
    echo "running push-api..."
    bash docker/api/push.sh "${DOCKERHUB_ACCOUNT}" "${DOCKERHUB_REPO}"
}

function push:migrate() {
    source .env
    echo "running push-migrate..."
    bash docker/migrate/push.sh "${DOCKERHUB_ACCOUNT}" "${DOCKERHUB_REPO}"
}

function push:inference() {
    source .env
    echo "running push-inference..."
    bash docker/inference/push.sh "${DOCKERHUB_ACCOUNT}" "${DOCKERHUB_REPO}"
}

function push:all() {
    echo "pushing all images..."
    push:api
    push:inference
    #push:migrate
}


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



function generate:pwd-hash() {
    # Check if passlib is already installed
    if ! python3 -c "import passlib" &> /dev/null; then
        # If passlib is not installed, install it
        pip install passlib
        PASSLIB_INSTALLED_BY_SCRIPT=true
    else
        PASSLIB_INSTALLED_BY_SCRIPT=false
    fi

    # Generate the hashed password using a Python script
    #SUPERUSER_PASSWORD="superuser_pwd"
    SUPERUSER_HASHED_PASSWORD=$(python3 -c "from passlib.context import CryptContext; pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto'); print(pwd_context.hash('$SUPERUSER_PASSWORD'))")

    # Update the .env file with the hashed password
    if grep -q "SUPERUSER_HASHED_PASSWORD" .env; then
        sed -i "s#^SUPERUSER_HASHED_PASSWORD=.*#SUPERUSER_HASHED_PASSWORD=$SUPERUSER_HASHED_PASSWORD#" .env
    else
        echo "SUPERUSER_HASHED_PASSWORD=$SUPERUSER_HASHED_PASSWORD" >> .env
    fi

    # Uninstall passlib if it was installed by this script
    if [ "$PASSLIB_INSTALLED_BY_SCRIPT" = true ]; then
        pip uninstall -y passlib
    fi
}

function verify:pwd-hash() {
    # The password to verify
    PASSWORD_TO_VERIFY="$1"
    # The hashed password
    HASHED_PASSWORD="$2"

    # Check if passlib is already installed
    if ! python3 -c "import passlib" &> /dev/null; then
        # If passlib is not installed, install it
        pip install passlib
    fi

    echo "Please note: If your password or hashed password contains special characters, they must be escaped with a backslash (\\). For example, if your password is pa\$\$word, you should enter it as pa\\\$\\\$word."    # Verify the password against the hash
    PASSWORD_MATCHES=$(python3 -c "from passlib.context import CryptContext; pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto'); print(pwd_context.verify('$PASSWORD_TO_VERIFY', '$HASHED_PASSWORD'))")

    echo "Password matches: $PASSWORD_MATCHES"
}

TIMEFORMAT="Task completed in %3lR"
time ${@:-help}



