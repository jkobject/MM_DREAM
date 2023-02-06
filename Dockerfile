FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install .
ENTRYPOINT ["python", "/app/mm_dream", "/app/data/MMpathways.gmt"]
