# Start from the CARLA Simulator image
FROM carlasim/carla:0.9.10.1

# Switch to root user
USER root

# Add the NVIDIA CUDA repository GPG key directly
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC


RUN packages='libsdl2-2.0 xserver-xorg libvulkan1' \
	&& apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y $packages --no-install-recommends \
    && VULKAN_API_VERSION=`dpkg -s libvulkan1 | grep -oP 'Version: [0-9|\.]+' | grep -oP '[0-9|\.]+'` && \
	mkdir -p /etc/vulkan/icd.d/ && \
	echo \
	"{\
		\"file_format_version\" : \"1.0.0\",\
		\"ICD\": {\
			\"library_path\": \"libGLX_nvidia.so.0\",\
			\"api_version\" : \"${VULKAN_API_VERSION}\"\
		}\
	}" > /etc/vulkan/icd.d/nvidia_icd.json \
	&& rm -rf /var/lib/apt/lists/*

#RUN apt-get update && apt-get install -y wget && \
#    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add -

# Install xdg-user-dirs
RUN apt-get update && \
    apt-get install -y xdg-user-dirs \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    x11-apps \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=carla:carla . /home/carla
# switch back to user "carla"
USER carla
WORKDIR /home/carla

# Set the environment variable
ENV SDL_VIDEODRIVER=offscreen
ENV DISPLAY=""

# Set the default command
CMD ["/bin/bash", "CarlaUE4.sh", "--world-port=2000", "-opengl"]
