FROM python:3.11

WORKDIR /app

RUN pip install --upgrade pip
RUN apt update

RUN pip install numpy  # Must precede GDAL

RUN apt install -y gdal-bin libgdal-dev
RUN pip install gdal==`gdal-config --version`

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py .
