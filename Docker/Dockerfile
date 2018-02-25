# Docker container for running my solutions to Udacity CS344 Intro to Parallel Programming code.

FROM nvidia/cudagl:9.0-devel-ubuntu16.04
LABEL maintainers="Tanmay Bangalore <tanmaybangalore@gmail.com>"

# Install OpenCV (https://github.com/BVLC/caffe/wiki/OpenCV-3.3-Installation-Guide-on-Ubuntu-16.04)
#RUN apt-get update && apt-get upgrade -y && \
#    apt-get install -y build-essential cmake git pkg-config unzip ffmpeg qtbase5-dev python-dev \
#    python3-dev python-numpy python3-numpy libopencv-dev libgtk-3-dev libdc1394-22 \
#    libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev \
#    libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev \
#    libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev \
#    libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev \
#    v4l-utils vtk6 liblapacke-dev libopenblas-dev libgdal-dev checkinstall wget unzip

#ENV HOME /home/nvidia
#RUN mkdir -p ${HOME} && cd ${HOME} && \
#    wget --output-document=opencv-3.3.0.zip --directory-prefix=${HOME} \
#    https://github.com/opencv/opencv/archive/3.3.0.zip && \
#    cd ${HOME} && unzip opencv-3.3.0.zip
#RUN cd ${HOME}/opencv-3.3.0 && mkdir build && cd build && \
#    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D FORCE_VTK=ON \
#    -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -D \
#    CUDA_NVCC_FLAGS="-D_FORCE_INLINES --expt-relaxed-constexpr" -D WITH_GDAL=ON -D WITH_XINE=ON \
#    -D BUILD_EXAMPLES=ON .. && \
#    make -j $(($(nproc) + 1))

RUN apt-get update && apt-get upgrade -y && apt-get install -y libopencv-dev python-opencv \
    build-essential cmake sudo konsole dbus-x11 oxygen-icon-theme* clang-format
RUN apt-get install -y --no-install-recommends cuda-samples-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

# Make a home directory so everything isn't just owned by root
ARG UID=1000
ARG GID=1000

RUN mkdir -p /home/nvidia && \
    echo "nvidia:x:${UID}:${GID}:nvidia,,,:/home/nvidia:/bin/bash" >> /etc/passwd && \
    echo "nvidia:x:${UID}:" >> /etc/group && \
    echo "nvidia ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/nvidia && \
    chmod 0440 /etc/sudoers.d/nvidia && \
    chown ${UID}:${GID} -R /home/nvidia
RUN mkdir -p /home/nvidia/.local/share/konsole
COPY konsole/* /home/nvidia/.local/share/konsole/
RUN chown -R ${UID}:${GID} /home/nvidia/.local/share/konsole/

USER nvidia
ENV HOME /home/nvidia