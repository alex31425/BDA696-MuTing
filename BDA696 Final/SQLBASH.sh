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

echo "commands completed"

mysql -h db -u root -proot baseball -e 'select * from data_final where home_team_id = 5636 and home_team_wins = 1 order by game_id,home_team_id;' > /results/results.txt

echo "check results"

python final.py

echo "python script ran"
