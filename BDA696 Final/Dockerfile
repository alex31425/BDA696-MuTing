FROM python:3.6

# LABEL about image
LABEL maintainer="mhuang7447@sdsu.edu"
LABEL description="BDA696"

RUN mkdir docker-build
WORKDIR /docker-build

RUN apt-get update && apt install mariadb-client -y
	
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
  && rm -rf /var/lib/apt/lists/*

# RUN pip3 install mysql-connector
# RUN pip3 install pandas	
RUN pip3 install Jinja2


COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt


COPY ./SQLBASH.sh /docker-build/SQLBASH.sh
COPY baseball.sql /docker-build/baseball.sql
COPY sql_commands.sql /docker-build/sql_commands.sql
COPY final.py /docker-build/final.py

RUN chmod +x SQLBASH.sh
RUN chmod +x final.py

CMD ["/docker-build/SQLBASH.sh"]
# CMD ["python","/docker-build/final.py"]

# https://www.geeksforgeeks.org/how-to-run-a-python-script-using-docker/
