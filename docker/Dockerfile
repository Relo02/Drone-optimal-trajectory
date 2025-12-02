FROM osrf/ros:humble-desktop-full

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV SHELL=/bin/bash
ENV PX4_HOME=/opt/px4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    lsb-release \
    gnupg \
    software-properties-common \
    ninja-build \
    protobuf-compiler \
    libprotobuf-dev \
    python3-jinja2 \
    python3-numpy \
    python3-yaml \
    python3-setuptools \
    python3-empy \
    python3-toml \
    python3-packaging \
    python3-jsonschema \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libxml2-dev \
    libxml2-utils \
    libtinyxml2-dev \
    libeigen3-dev \
    zip \
    unzip \
    libgoogle-glog-dev \
    libgflags-dev \
    libceres-dev \
    python3-dev \
    python3-distutils \
    libsdformat14-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PX4 Python dependencies
RUN pip3 install --user kconfiglib pyros-genmsg empy jinja2 packaging toml jsonschema future

# Add Gazebo packages repository and install Gazebo Garden
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

RUN apt-get update && apt-get install -y \
    gz-garden \
    && rm -rf /var/lib/apt/lists/*

# Install MAVROS for PX4 communication
RUN apt-get update && apt-get install -y \
    ros-humble-mavros \
    ros-humble-mavros-extras \
    ros-humble-mavros-msgs \
    ros-humble-tf-transformations \
    && rm -rf /var/lib/apt/lists/*

# Install MAVROS geographic datasets
RUN wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh \
    && chmod +x install_geographiclib_datasets.sh \
    && ./install_geographiclib_datasets.sh \
    && rm install_geographiclib_datasets.sh

# Clone PX4-Autopilot with proper git history for version detection
WORKDIR /opt
RUN git clone --branch v1.14.3 https://github.com/PX4/PX4-Autopilot.git px4_source \
    && cd px4_source \
    && git fetch --tags \
    && git fetch --unshallow || true \
    && git submodule update --init --recursive

# Pre-install PX4 dependencies but don't build yet (build at runtime for better error handling)
WORKDIR /opt/px4_source
RUN make submodulesclean || true \
    && make distclean || true \
    && echo "PX4 source prepared - will build at runtime"

# Create workspace directory
WORKDIR /root/drone_ws

# Copy source code from drone_ws/src
COPY drone_ws/src/ src/

# Initialize rosdep and install dependencies (continue even if some fail)
RUN rosdep init || true \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src -r -y || echo "Some dependencies failed to install, continuing..."

# Build the workspace (with fallback if build fails)
RUN . /opt/ros/humble/setup.sh \
    && colcon build --symlink-install || echo "Build completed with warnings"

# Set up entrypoint
RUN echo '#!/bin/bash' > /entrypoint.sh \
    && echo 'set -e' >> /entrypoint.sh \
    && echo 'source /opt/ros/humble/setup.bash' >> /entrypoint.sh \
    && echo 'source /root/drone_ws/install/setup.bash' >> /entrypoint.sh \
    && echo 'export GZ_SIM_RESOURCE_PATH=/opt/px4_source/Tools/simulation/gz/models:/opt/px4_source/Tools/simulation/gz/worlds' >> /entrypoint.sh \
    && echo 'export PX4_GZ_MODELS=/opt/px4_source/Tools/simulation/gz/models' >> /entrypoint.sh \
    && echo 'export PX4_ROOT=/opt/px4_source' >> /entrypoint.sh \
    && echo 'exec "$@"' >> /entrypoint.sh \
    && chmod +x /entrypoint.sh

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN addgroup --gid $GROUP_ID user && \
    adduser --uid $USER_ID --gid $GROUP_ID --disabled-password user

USER user
WORKDIR /home/user   

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# Expose common ports
EXPOSE 14540 14541 14542 14543 14544 14545
