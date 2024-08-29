FROM python:3.9.5

WORKDIR /app

ADD . /app

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN set -ex; \
    pip install -r requirements.txt --default-timeout=100000 --no-compile;
