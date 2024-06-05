# file: app/database.py
"""Module: database.py.

This module provides the database configuration and setup for the application.
It uses SQLAlchemy as the ORM (Object-Relational Mapping) library to interact
with the database.

The module includes the following main components:

`SQLALCHEMY_DATABASE_URL`: The database connection URL obtained from the env variable "SQL_URL".
SQLALCHEMY_DATABASE_URL = os.environ["SQL_URL"]
- `engine`: The SQLAlchemy engine instance created based on the database URL.
- `SessionLocal`: The sessionmaker factory used to create database sessions.
- `Base`: The declarative base class for defining database models.
- `get_db`: A dependency function that yields a database session for each request.

To use a different database system (e.g., PostgreSQL, MySQL), uncomment the appropriate
`create_engine` line and modify the connection URL accordingly.

When using SQLite, the `connect_args` parameter is set to `{"check_same_thread": False}`
to allow multiple threads to access the database concurrently.

The `get_db` function is used as a dependency in FastAPI routes to provide a database
session to the route handlers. It ensures that the session is properly closed after
each request.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Get the environment variables
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

# Construct the database URL
DATABASE_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URI)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()