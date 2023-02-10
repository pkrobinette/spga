#
# Self-Preserving Genetic Algorithms
#

FROM python:3.9

# set the user to root
USER root

# set working directory to spga
WORKDIR /spga

# install pip
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip
RUN pip install --upgrade pip

# set environment variables
ENV PYTHONPATH='/spga'
# ENV OPENBLAS_NUM_THREADS=1
# ENV OMP_NUM_THREADS=1

# copy files to docker
COPY . .

# install python package dependencies
RUN pip install .


# FROM continuumio/miniconda3

# # Create the environment:
# COPY environment.yml .

# # Demonstrate the environment is activated:
# RUN echo "Made it!"
# RUN python -c "import flask"

# The code to run when container is started:
# COPY run.py entrypoint.sh ./
# ENTRYPOINT ["./entrypoint.sh"]

# # Create the environment:
# COPY environment.yml .
# RUN conda env create -f environment.yml

# # Make RUN commands use the new environment:
# # Pull the environment name out of the environment.yml
# RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
# ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

# # Demonstrate the environment is activated:
# RUN echo "Make sure flask is installed:"
# RUN python -c "import flask"



# FROM python:3.9

# # set the user to root
# USER root

# # set working directory to spga
# WORKDIR /spga

# # install pip
# RUN set -xe \
#     && apt-get update -y \
#     && apt-get install -y python3-pip
# RUN pip install --upgrade pip

# # set environment variables
# ENV PYTHONPATH='/spga'
# # ENV OPENBLAS_NUM_THREADS=1
# # ENV OMP_NUM_THREADS=1

# # copy files to docker
# COPY . .

# # install python package dependencies
# RUN pip install -r requirements.txt