# Use the official Python image with Python 3.11
FROM python:3.11-slim

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

# Command to run the Streamlit application
CMD ["streamlit", "run", "neuraview/main.py"]