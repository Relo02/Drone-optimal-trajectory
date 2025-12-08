FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true
#ENV DISPLAY=:99
ENV LANG=C.UTF-8

RUN apt-get update \
    && apt-get install -y \
       cmake \
       curl \
       git \
       python3-pip \
       lsb-release \
       sudo \
       tmux \
       wget \
       nano
RUN pip3 install --break-system-packages pyros-genmsg
RUN git clone https://github.com/PX4/PX4-Autopilot.git --recursive \
    && bash ./PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx
# ENV QT_X11_NO_MITSHM=1

RUN curl -O https://d176tv9ibo4jno.cloudfront.net/builds/master/QGroundControl-x86_64.AppImage \
    && chmod +x QGroundControl-x86_64.AppImage \
    && sudo usermod -aG dialout "$(id -un)" \
    && mkdir QGC \
    && mv QGroundControl-x86_64.AppImage QGC

WORKDIR "/QGC"
RUN ./QGroundControl-x86_64.AppImage --appimage-extract 

RUN apt-get -y autoremove && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

WORKDIR /PX4-Autopilot

RUN echo "ubuntu ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers && \
    chmod 0440 /etc/sudoers && \
    chmod g+w /etc/passwd 
