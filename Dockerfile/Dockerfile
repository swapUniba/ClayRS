ARG VERSION=3.8
FROM python:$VERSION

COPY . /usr/src/repo

WORKDIR /usr/src/repo
RUN pip install wheel sphinx lenskit && python setup.py install

WORKDIR /usr/src
RUN rm -rf repo
