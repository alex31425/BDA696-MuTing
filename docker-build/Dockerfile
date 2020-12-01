FROM ubuntu

# LABEL about image
LABEL maintainer="mhuang7447@sdsu.edu"
LABEL description="BDA696"

RUN apt-get update && apt install mariadb-client -y

RUN mkdir docker-build
WORKDIR /docker-build

COPY ./SQLBASH.sh /docker-build/SQLBASH.sh
COPY baseball.sql /docker-build/baseball.sql
COPY sql_commands.sql /docker-build/sql_commands.sql

RUN chmod +x SQLBASH.sh

CMD ["/docker-build/SQLBASH.sh"]


