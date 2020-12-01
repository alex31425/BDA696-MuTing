#!/bin/bash

sleep 10

# Insert the SQL database (if it doesn't exist)
if ! mysql -h db -uroot -proot -e 'use baseball'; then
  echo "Baseball DOES NOT exists"
  mysql -h db -proot -u root -e "CREATE DATABASE IF NOT EXISTS baseball;"
  mysql -h db -u root -proot baseball < /docker-build/baseball.sql
else
  echo "Baseball DOES exists"
fi

# Run the sql commands
mysql -h db -u root -proot baseball < /docker-build/sql_commands.sql

mysql -h db -u root -proot baseball -e '
  SELECT batter,sum(HIT)/sum(ATBAT) as AVG
  FROM rolling group by batter order by batter, sum(HIT)/sum(ATBAT);' > /results/results.txt
