# file: app/main.py
"""Module: main.py.

This module serves as the entry point for the FastAPI application. It sets up the
API routes, database connection, and handles the application's lifespan events.

The module includes the following main components:


- API:
  - The FastAPI application instance is created.
  - API routes are included using the `include_router` method for different modules:
    - `auth`: Authentication-related routes.
    - `admin`: Admin-related routes.
    - `users`: User-related routes.
    - `bert_sentiment`: Machine learning service BERT sentiment analyzer routes.
    - `roberta_emotion`: Machine learning service roBERTa emotion analyzer analyzer routes.

- Lifespan Events:
  - `startup_event`: Called when the application starts up. It creates a superuser if the
    environment variable "CREATE_SUPERUSER" is set to a truthy value. It also loads the
    machine learning models.
  - `shutdown_event`: Called when the application shuts down. It removes the superuser if
    the environment variable "CREATE_SUPERUSER" is set to a truthy value.

- Routes:
  - `/healthcheck`: A simple healthcheck endpoint to check the status of the application.

The module loads environment variables from the `.environment` folder using the `load_dotenv`
function from the `python-dotenv` library.

Note: Make sure to have the necessary environment variables set up in the `.environment`
folder for the application to function properly.
"""

import os

from fastapi import FastAPI

from database import SessionLocal, engine

from routers import admin, auth, users#, bert_sentiment, roberta_emotion, users
from loguru import logger

#load_dotenv(override=True)  # loads environment variables from the .environment folder

############### API ###############
logger.info("Starting FastAPI...")
app = FastAPI()
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(users.router)
# app.include_router(bert_sentiment.router)
# app.include_router(roberta_emotion.router)
logger.info("All FastAPI routers added")


############### ROUTES ###############
@app.get("/healthcheck")
def get_healthcheck():
    """Healthcheck endpoint to check the status of the application.

    This function is used to check the health status of the FastAPI application. It returns a JSON
    response indicating that the application is healthy.

    Returns
    -------
        dict: A dictionary containing the health status of the application.
            - "status" (str): The status of the application, which is always "healthy".

    Raises
    ------
        None

    """
    return {"status": "healthy"}
