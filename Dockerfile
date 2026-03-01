# This Dockerfile was created with the help of https://pythonspeed.com/articles/activate-conda-dockerfile/
# I changed it to work with my needs. Note this docker only needs to be created once and then the code clones it for interference.
FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yaml /tmp/environment.yaml
RUN conda env create -f /tmp/environment.yaml

# Add the environment activation command to the default bash profile
RUN echo "conda activate pyenv-gpu" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Default command to keep container alive (overwritten when code is run)
CMD ["tail", "-f", "/dev/null"]
