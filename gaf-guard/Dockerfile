FROM python:3.12.4

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Set env variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch to the "user" user
USER user

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy requirements.txt and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY --chown=user src/ .

# Set PythonPath to source directory
ENV PYTHONPATH="$PYTHONPATH:$HOME/app"