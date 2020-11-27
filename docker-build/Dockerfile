FROM ubuntu

# LABEL about image
LABEL maintainer="mhuang7447@sdsu.edu"
LABEL description="BDA696"

RUN apt-get update && apt install mariadb-client -y

WORKDIR /home/mu-ting/docker-build

COPY ./SQLBASH.sh /home/mu-ting/docker-build/SQLBASH.sh
COPY baseball.sql /home/mu-ting/docker-build/baseball.sql

RUN chmod +x SQLBASH.sh


VOLUME ["/home/mu-ting/docker-build","SQLBASH.sh","baseball.sql"]

CMD ["/home/mu-ting/docker-build","baseball.sql","SQLBASH.sh"]


