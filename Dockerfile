# Use an official Python runtime as a parent image
FROM python:3.12.6

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
ADD . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]