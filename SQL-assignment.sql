use baseball;

# create a table that has a column called DATEFIDD whcih calculate
# the differecnes day between a specific date and local_date in game table
drop table if exists batting_avg_wDATEDIFF ;
CREATE table batting_avg_wDATEDIFF (
batter int unsigned not null,
hit int unsigned not null,
atbat int unsigned not null,
local_date datetime not null,
diff_day int not null
) engine=InnoDB default charset latin1;


# Insert data into batting_avg_wDATEDIFF table
insert into batting_avg_wDATEDIFF (
batter,hit,atbat,local_date,diff_day)
select bc.batter,bc.hit,bc.atBat,g.local_date,DATEDIFF('2012-07-05 00:00:00' ,g.local_date) 
from game g
join batter_counts bc on
	bc.game_id = g.game_id;

select * from batting_avg_wDATEDIFF bawd ;
DELETE from batting_avg_wDATEDIFF ;

# select data from batting_avg_wDATEDIFF table that DATEDIFF is fewer than 100 days	
set @Enddate = '2008-08-31';

drop table if exists rolling;
CREATE table rolling
select a.batter ,a.local_date as Beg_date ,b.local_date as End_date, b.hit as HIT,b.atbat as ATBAT, DATEDIFF(b.local_date ,a.local_date) as date_diff
from batting_avg_wDATEDIFF a inner join batting_avg_wDATEDIFF b on a.batter = b.batter 
where b.local_date < @Enddate order by a.local_date, DATEDIFF(b.local_date ,a.local_date) ;

select * from rolling ;
select batter,Beg_date,SUM(HIT),SUM(ATBAT),ROUND(SUM(HIT)/SUM(ATBAT),3) as AVG from rolling where date_diff between 0 AND 100 and Beg_date > SUBDATE(@Enddate,INTERVAL 100 DAY) group by Beg_date,batter order by batter,Beg_date;

DELETE from rolling;

# select data from batting_avg_wDATEDIFF table that DATEDIFF is 365 days (1 year)

select batter as batter_365 ,SUM(hit),SUM(atbat),SUM(hit)/SUM(atbat) as AVG from batting_avg_wDATEDIFF 
where diff_day between 0 and 365
group by batter
order by AVG desc;

# create a table to store the above result
drop table if exists batting_avg_365;
CREATE table batting_avg_365 (
batter int unsigned not null,
hit int unsigned not null,
atbat int unsigned not null,
AVG float unsigned
) engine=InnoDB default charset latin1;

# store data in the batting_avg_365 table
insert into batting_avg_365 (batter,hit,atbat,AVG)
select batter ,SUM(hit),SUM(atbat),
CASE when SUM(hit)<>0
then round(SUM(hit)/SUM(atbat),3)
else 0 end as AVG
from batting_avg_wDATEDIFF 
where diff_day between 0 and 365
group by batter
order by AVG desc;

select batter,hit,atbat,ROUND(AVG,3) as AVG from batting_avg_365 order by batter ;
DELETE from batting_avg_365;


# Every player's alltime battering average
select batter, sum(Hit) as TH,sum(atBat) as TB, SUM(Hit)/SUM(atBat) as AVG from batter_counts group by batter;

# create a table to store above data
drop table if exists batting_avg_alltime;
CREATE table batting_avg_alltime (
batter int unsigned not null,
hit int unsigned null,
atbat int unsigned not null,
AVG float unsigned
) engine=InnoDB default charset latin1;

select * from batting_avg_alltime ;

# insert data into batting_avg_alltime table
insert into batting_avg_alltime (batter,hit,atbat,AVG)
select batter, SUM(Hit),sum(atBat),
case when SUM(Hit)<>0 then round(SUM(Hit)/SUM(atBat),3)
else 0 end as AVG
from batter_counts group by batter;

select batter,hit,atbat,round(AVG,3) as AVG from batting_avg_alltime ;
DELETE from batting_avg_alltime ;


