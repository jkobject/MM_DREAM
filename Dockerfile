FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install .
#COPY ./cmd.sh /
#RUN chmod 755 /cmd.sh
ENTRYPOINT ["owkin_mm_dream"]
#CMD ["/cmd.sh"]