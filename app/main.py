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
- Database:
  - The database tables are created based on the defined models using `Base.metadata.create_all`.

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

from dotenv import load_dotenv
from fastapi import FastAPI

from .database import SessionLocal, engine
from .devtools import create_superuser, remove_superuser
from .ml_models.bert_sentiment import BERTSentimentAnalyzer
from .ml_models.roberta_emotion import RobertaEmotionAnalyzer
from .models import Base
from .routers import admin, auth, bert_sentiment, roberta_emotion, users

load_dotenv(override=True)  # loads environment variables from the .environment folder

############## ML MODELS ###############
bert_model = BERTSentimentAnalyzer()
roberta_model = RobertaEmotionAnalyzer()

############### API ###############
app = FastAPI()
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(users.router)
app.include_router(bert_sentiment.router)
app.include_router(roberta_emotion.router)


############### DATABASE ###############
Base.metadata.create_all(bind=engine)


############### LIFESPAN ###############
@app.on_event("startup")
async def startup_event():
    """Startup event handler for the FastAPI application.

    This function is called when the application starts up. It performs the following tasks:
    1. If the environment variable "CREATE_SUPERUSER" is set to a truthy value,
    it creates a superuser in the database using the `create_superuser` function.
    2. Loads the machine learning models for version 1 and version 2 using the `load_model` method
       of the respective model instances.

    Returns
    -------
        None

    Raises
    ------
        None

    """
    if os.getenv("CREATE_SUPERUSER", "False").lower() in ["true", "1", "yes"]:
        db = SessionLocal()
        create_superuser(db)
        db.close()

    await bert_sentiment.bert_sentiment_analyzer.load_model()
    await roberta_emotion.roberta_emotion_analyzer.load_model()


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler for the FastAPI application.

    This function is called when the application shuts down. It performs the following task:
    1. If the environment variable "CREATE_SUPERUSER" is set to a truthy value,
    it removes the superuser from the database using the `remove_superuser` function.

    Returns
    -------
        None

    Raises
    ------
        None

    """
    if os.getenv("CREATE_SUPERUSER", "False").lower() in ["true", "1", "yes"]:
        db = SessionLocal()
        remove_superuser(db)
        db.close()


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
