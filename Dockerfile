FROM python:3.13.2-bookworm

# Set the working directory inside the container
WORKDIR /workspace

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the entrypoint to run main.py
CMD ["python", "main.py"]