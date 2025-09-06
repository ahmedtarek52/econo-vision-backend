# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies from requirements.txt
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Tell the container to listen on port 8080
EXPOSE 8080

# The command to run your app using Gunicorn, a production-ready server
CMD ["gunicorn", "--bind", ":8080", "--workers", "2", "app:create_app()"]

