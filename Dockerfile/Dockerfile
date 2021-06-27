ARG VERSION=3.8
FROM python:$VERSION

COPY setup.py /usr/src/

WORKDIR /usr/src/
RUN pip install -e . && rm setup.py
