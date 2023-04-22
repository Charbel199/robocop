FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y libpython3.8-stdlib
RUN apt-get install -y python3.8 python3-pip python3.8-dev
RUN apt-get install -y libusb-1.0.0

RUN pip3 install opencv-python numpy

# This package requires user input, install manually:
# RUN apt-get install -y libqt5widgets5 python3-tk

ENV QT_QPA_PLATFORM=xcb
ENV PYTHONPATH=/home/rover

WORKDIR /home/rover
CMD ["bash"]
