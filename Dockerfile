# FROM    pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
FROM legosz/jojogan:v1

WORKDIR /app
# COPY    . .

RUN pip install --upgrade pip
RUN pip install cmake
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE  5000
CMD ["python3", "main.py"]