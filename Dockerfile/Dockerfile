ARG VERSION=3.8
FROM python:$VERSION

COPY . /usr/src/repo

WORKDIR /usr/src/repo
RUN pip install wheel sphinx lenskit

WORKDIR /usr/src
RUN rm -rf repo
