FROM ros:noetic

RUN apt-get update

RUN apt-get install -y python3-pip
RUN apt-get install -y libusb-1.0.0

RUN pip3 install --upgrade numpy==1.20
RUN pip3 install opencv-python
RUN pip3 install scikit-fuzzy matplotlib RPi.GPIO

RUN apt-get install -y libqt5widgets5 python3-tk

ENV QT_QPA_PLATFORM=xcb
ENV PYTHONPATH=/home/rover

COPY . /home/rover
WORKDIR /home/rover

RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_make'

RUN echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
RUN echo 'source /home/rover/devel/setup.bash' >> ~/.bashrc

CMD ["bash"]
