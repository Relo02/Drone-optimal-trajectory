FROM osrf/ros:humble-desktop
RUN curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py \
    && pip install -U pip setuptools wheel packaging \
    && pip install -U empy==3.3.4 pyros-genmsg
RUN git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
WORKDIR "/Micro-XRCE-DDS-Agent"
RUN mkdir build
WORKDIR "/Micro-XRCE-DDS-Agent/build"
RUN cmake ..
RUN make
RUN sudo make install
RUN sudo ldconfig /usr/local/lib/

RUN apt-get update && \
    apt-get install -y tmux

RUN apt-get -y autoremove && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*
