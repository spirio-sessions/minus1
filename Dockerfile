# Use the official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project directory into the container at /app
COPY . /app

# Install any needed packages specified in requirements_tensorflow_deprecated.txt
RUN pip install --no-cache-dir -r requirements_tensorflow_deprecated.txt

# Expose any required ports (e.g., for Jupyter Notebook)
EXPOSE 8888

# Set a default command that can be overridden
CMD ["bash"]
