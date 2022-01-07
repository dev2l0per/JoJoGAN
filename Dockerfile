# FROM    pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM legosz/jojogan:v1

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app
# COPY    . .

RUN apt-get update && apt-get install -y \
  build-essential \
  libgl1-mesa-glx \
  libglib2.0-0
  
ENV CUDA_HOME="/usr/local/cuda-10.2"
ENV PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

RUN pip3 install cmake
RUN pip3 install dlib

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE  5000
CMD ["python3", "main.py"]