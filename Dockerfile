# FROM    pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
FROM legosz/jojogan:v1

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app
# COPY    . .

RUN apt-get update && apt-get install -y \
  tzdata \
  build-essential \
  libgl1-mesa-glx \
  libx11-dev \
  # ffmpeg \
  # libsm6 \
  # libxext6 \
  python3-opencv \
  cmake \
  git

RUN git config --global http.postBuffer 1048576000

RUN mkdir -p dlib && \
    git clone https://github.com/davisking/dlib.git dlib/ && \
    cd dlib/ && \
    mkdir build; cd build; cmake ..; cmake --build .
    # cd .. \
    # python3 setup.py install && \
    # cd /app

RUN pip3 install --upgrade pip
# RUN pip3 install cmake
# RUN pip3 install ./dlib-19.17.0-cp37-cp37m-win_amd64.whl
# RUN pip uninstall cmake dlib
# RUN pip install -U wheel cmake
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE  5000
CMD ["python3", "main.py"]