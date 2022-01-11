# docker build -t height_clf:v0.0.1 .
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install required apt packages
RUN apt-get update && apt-get install -y build-essential libsndfile1 git sox wget \
&& rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
ADD requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN export GIT_TOKEN="DUMMY TOKEN" \
    && python3 -m pip install --no-cache-dir git+https://$GIT_TOKEN@github.com/DinoHub/s3u.git \
    # && python3 -m pip install --no-cache-dir git+https://$GIT_TOKEN@github.com/DinoHub/augdino.git \
    && unset GIT_TOKEN

WORKDIR /clf
RUN ["bash"]