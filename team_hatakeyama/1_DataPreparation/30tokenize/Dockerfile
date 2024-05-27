FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

LABEL maintainer="ssone"

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# User Settings & Installation of Required Packages
USER root

RUN apt update && \
    apt install -y bash python3 python3-pip && \
    apt install -y libglew-dev ca-certificates zip unzip bzip2 lsof less nkf swig && \
    apt install -y x11-xserver-utils xvfb pkg-config libsdl2-2.0-0 libsdl2-dev && \
    apt install -y libgtk-3-dev libgstreamer-gl1.0-0 && \
    apt install -y libhdf5-dev libfreetype6-dev && \
    apt install -y libsdl1.2-dev libsdl-image1.2-dev libsdl-ttf2.0-dev && \
    apt install -y libsdl-mixer1.2-dev libportmidi-dev libx264-dev

RUN apt install -y sudo curl wget ssh vim emacs git gcc g++ make cmake git-lfs && \
    apt install -y tesseract-ocr espeak-ng ca-certificates bzip2 zip && \
    apt install -y tree htop bmon iotop build-essential openjdk-8-jdk-headless gfortran && \
    apt install -y build-essential x11-apps libaio-dev && \
    apt install -y emacs nkf graphviz graphviz-dev && \
    apt install -y language-pack-ja-base language-pack-ja && \
    apt install -y mecab libmecab-dev mecab-ipadic-utf8 && \
    apt clean && rm -rf /var/lib/apt/lists/*


##########################################################################################
# root password
RUN echo 'root:hanako' | chpasswd


# Create a user named "ssone" and set up the home directory.
ARG USERNAME=ssone
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# ユーザーとグループを作成し、sudoをインストール
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Minicondaのインストール
USER $USERNAME
WORKDIR /home/$USERNAME
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm Miniconda3-latest-Linux-x86_64.sh

# PATHの設定
ENV PATH /home/$USERNAME/miniconda3/bin:$PATH

# Python-related setupls
COPY requirements_nedo2024.txt .
# Add conda configuration to .bashrc
RUN echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc && \
    echo 'export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"' >> ~/.bashrc && \
    echo 'source activate base' >> ~/.bashrc

# Install Jupyter notebook
RUN $HOME/miniconda3/bin/conda install -c conda-forge jupyter -y


RUN conda create -n scr -y python=3.11.2 \
    && conda install -n scr ipykernel \
    && /home/$USERNAME/miniconda3/envs/scr/bin/python -m ipykernel install --user --name scr --display-name "Python3.11.2 (scr)" \
    && /home/$USERNAME/miniconda3/envs/scr/bin/python -m pip install -r requirements_nedo2024.txt \
    && /home/$USERNAME/miniconda3/envs/scr/bin/python -m pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html \
    && /home/$USERNAME/miniconda3/envs/scr/bin/python -m pip install deepspeed-kernels \
    && DS_BUILD_OPS=1 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_SPARSE_ATTN=0 /home/$USERNAME/miniconda3/envs/scr/bin/python -m pip install deepspeed==0.12.4 \
    && /home/$USERNAME/miniconda3/envs/scr/bin/python -m pip install unidic \
    && /home/$USERNAME/miniconda3/envs/scr/bin/python -m unidic download

##########
# Add to Jupyter's path
RUN echo 'export PATH=$PATH:/home/$USERNAME/.local/bin' >> ~/.bashrc

# Setting Jupyter Notebook (Allow access Without password)
RUN mkdir -p /home/$USERNAME/.jupyter && \
    echo "c.NotebookApp.token = ''\nc.NotebookApp.password = ''" > /home/$USERNAME/.jupyter/jupyter_notebook_config.py

#WORKDIR /home/$USERNAME/.ssh
#COPY authorized_keys authorized_keys

########################################################################################
USER root

RUN apt update && \
    apt install -y ffmpeg

# Open SSH Port
RUN mkdir /var/run/sshd
EXPOSE 22

# Open Jupyter Notebook Port
EXPOSE 8888

# Setting Env
#ENV DISPLAY host.docker.internal:0.0
# ENV OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

USER $USERNAME
WORKDIR /home/

CMD ["/usr/sbin/sshd", "-D"]
SHELL ["/bin/bash", "-c"]