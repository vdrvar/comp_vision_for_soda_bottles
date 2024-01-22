# Use the official Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container image
COPY . /app

# Install system dependencies for Pillow
RUN apt-get update && apt-get install -y libjpeg-dev zlib1g-dev libtiff5-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Command to run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "predict_bottles:app"]
