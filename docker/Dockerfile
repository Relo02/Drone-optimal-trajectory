FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install basic dependencies and locale support
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    locales \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    software-properties-common \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Add ROS 2 apt repository
RUN add-apt-repository universe \
    && apt-get update \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
       > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 Humble
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    ros-humble-desktop \
    ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Clone PX4-Autopilot with submodules
RUN git clone https://github.com/PX4/PX4-Autopilot.git --recursive

# Run PX4 setup script
RUN cd PX4-Autopilot/Tools/setup && bash ./ubuntu.sh

# Build px4_sitl
RUN cd PX4-Autopilot && make px4_sitl

# --------------------------------------------------------------------
# Install Micro-XRCE-DDS-Agent (v2.4.3)
# --------------------------------------------------------------------
RUN git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
    && cd Micro-XRCE-DDS-Agent \
    && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig /usr/local/lib/

# --------------------------------------------------------------------
# Create non-root user
# --------------------------------------------------------------------
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN addgroup --gid $GROUP_ID user && \
    adduser --uid $USER_ID --gid $GROUP_ID --disabled-password --gecos "" user

# Install Python dependencies for the user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
RUN pip install --user -U empy==3.3.4 pyros-genmsg setuptools

WORKDIR /home/user

# Source ROS 2 for the user
RUN echo "source /opt/ros/humble/setup.bash" >> /home/user/.bashrc

# Copy entrypoint and set permissions
COPY entrypoint.sh /home/entrypoint.sh

USER root
RUN chown -R user:user /home/user /home/entrypoint.sh \
    && chmod 755 /home/entrypoint.sh

USER user

ENTRYPOINT ["/home/entrypoint.sh"]
CMD ["bash"]

# Expose MAVLink ports
EXPOSE 14540 14541 14542 14543 14544 14545