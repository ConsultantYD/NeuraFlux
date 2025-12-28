# Use the official Python image with Python 3.11
FROM python:3.11-slim

# Install git (VS Code source control needs the git CLI in the container)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install poetry
RUN pip install poetry

# Disable virtual environments creation by poetry, install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install

# Expose the port Streamlit runs on
EXPOSE 8501