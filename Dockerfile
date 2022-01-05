FROM python:3.8 as builder

RUN mkdir /install
WORKDIR /install

COPY requirements.txt /requirements.txt

RUN pip install --prefix=/install -r /requirements.txt

FROM python:3.8-slim

ARG version_number
ARG commit_sha

ENV VERSION_NUMBER=$version_number
ENV COMMIT_SHA=$commit_sha
ENV test=test

COPY --from=builder /install /usr/local
COPY CRIMAC_threshold_classifier.py /app/CRIMAC_threshold_classifier.py
COPY config.json /app/config.json

WORKDIR /app

CMD ["python3", "/app/CRIMAC_threshold_classifier.py"]
