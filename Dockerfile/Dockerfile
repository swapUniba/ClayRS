ARG VERSION=3.8
FROM python:$VERSION

COPY requirements.txt /usr/src/

WORKDIR /usr/src/
RUN pip install -r requirements.txt && rm requirements.txt
