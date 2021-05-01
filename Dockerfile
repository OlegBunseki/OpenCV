FROM ubuntu:18.04

######################################################################
### BASE IMAGE
######################################################################

ARG CONDA_VERSION=4.9.2
ARG PYTHON_VERSION=3.7

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/miniconda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y \
        wget \ 
        bzip2 \
        fuse \
        ffmpeg libsm6 libxext6 \ 
        cmake build-essential libgtk-3-dev libboost-all-dev \
        libqt5dbus5 libqt5x11extras5 \
        && \
    # apt-get --reinstall install libqt5dbus5 \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# RUN apt-get --reinstall install libqt5dbus5 \
#         libqt5widgets5 libqt5network5 libqt5gui5 libqt5core5a \
#         libdouble-conversion1 libxcb-xinerama0

# RUN useradd --create-home dockeruser
# WORKDIR /home/dockeruser
# USER dockeruser

######################################################################
### CUSTOM FOLDER FOR THE PROJECT
######################################################################

RUN mkdir opencv
RUN mkdir opencv/tmp
COPY . localfiles


######################################################################
### INSTALL ANACONDA
######################################################################

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_${CONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh && \
    ~/miniconda/bin/conda clean -tipsy

ENV PATH="root/miniconda/bin/:${PATH}"

RUN conda install -y conda=${CONDA_VERSION} python=${PYTHON_VERSION} && \
    conda clean -aqy && \
    rm -rf ~/miniconda/pkgs && \
    find ~/miniconda/ -type d -name __pycache__ -prune -exec rm -rf {} \;

# RUN xhost +

######################################################################
### CUSTOM ANACONDA ENVIRONMENT
######################################################################

COPY conda_dependencies.yml opencv/tmp/conda_dependencies.yml

ENV PYTHONENV=opencv
RUN conda env create -n ${PYTHONENV} -f opencv/tmp/conda_dependencies.yml

### ADD CUSOTM ENVIRONMENT TO PATH
ENV PATH="root/miniconda/envs/${PYTHONENV}/bin/:${PATH}"


######################################################################
### SOME TESTS
######################################################################

# RUN conda init bash -> This will activate base Environment on Docker image

RUN echo "source activate ${PYTHONENV}" > ~/.bashrc


######################################################################
### RUN DOCKER
######################################################################

COPY entrypoint.sh opencv/tmp/entrypoint.sh
ENTRYPOINT [ "./opencv/tmp/entrypoint.sh" ]
CMD [ "default" ]