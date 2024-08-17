# Use an official Python runtime as the base image
FROM python:3.10-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the requirements file into the container
COPY requirements.txt .

# Install the project dependencies
RUN pip install -r requirements.txt

# Copy the rest of the project code into the container
COPY . .

# Set the command to run when the container starts
CMD [ "python", "main.py" ]