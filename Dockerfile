# FROM    pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM legosz/jojogan:v1

ARG DEBIAN_FRONTEND=noninteractive
# ENV TZ=Asia/Seoul
ENV CUDA_HOME="/usr/local/cuda-10.2"
# ENV PATH="/usr/local/cuda-10.2/bin${PATH:+:${PATH}}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

WORKDIR /app
# COPY    . .

RUN apt-get update && apt-get install -y \
#   tzdata \
#   build-essential \
  libgl1-mesa-glx \
  libglib2.0-0
# apt-get install libglib2.0-0
#   libx11-dev \
#   python3-opencv \
#   cmake \
#   git

# RUN . /root/.bashrc && \
#     conda init bash && \
#     conda update -n base -c defaults conda && \
#     conda create --name jojogan python=3.7 && \
#     conda activate jojogan && \
#     conda install -c conda-forge dlib

# RUN git config --global http.postBuffer 1048576000

# RUN mkdir -p dlib && \
#     git clone https://github.com/davisking/dlib.git dlib/ && \
#     cd dlib/ && \
#     mkdir build; cd build; cmake ..; cmake --build .

RUN pip3 install cmake
RUN pip3 install dlib

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE  5000
CMD ["python3", "main.py"]