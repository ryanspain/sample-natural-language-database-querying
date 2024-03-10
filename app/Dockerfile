FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Assuming the pip requirements will rarely change, we'll cache them in the images
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp
