FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install .
COPY ./cmd.sh /
RUN chmod +x /cmd.sh
ENTRYPOINT ["/cmd.sh"]