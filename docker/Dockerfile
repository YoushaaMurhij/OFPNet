FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install required libs and packages
RUN apt-get update -y && \
    apt-get autoremove -y && \ 
    apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ca-certificates  lsb-release build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Update pip & install pytorch/ pip pkgs
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install numpy matplotlib tqdm waymo_open_dataset_tf_2_6_0==1.4.3

# Install latest pytorch
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip3 install wandb timm IPython

ARG WORKSPACE
# COPY . ${WORKSPACE}

ENV PYTHONPATH "${PYTHONPATH}:${WORKSPACE}/"
WORKDIR ${WORKSPACE}

# Add docker user
# ARG USER=docker_occupancy_flow
# ARG UID=1001
# ARG GID=1001
# # default password
# ARG PW=user 
# RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo

# WORKDIR /home/${USER}
# RUN cd /home/${USER} && mkdir workspace && chown -R ${UID}:${GID} ./

# USER ${UID}:${GID}
