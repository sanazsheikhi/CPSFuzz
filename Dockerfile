FROM ubuntu:focal
MAINTAINER Edward Kim EMAIL ehkim@cs.unc.edu
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y python3 python3-pip git

RUN git clone https://github.com/f1tenth/f1tenth_gym.git
RUN cd f1tenth_gym
RUN git checkout exp_py
RUN pip3 install --user -e gym/

RUN git clone git@github.com:sanazsheikhi/CPSFuzz.git
RUN cd CPSFuzz
RUN pip3 install -r requirements.txt


