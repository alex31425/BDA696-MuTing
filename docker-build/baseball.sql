use baseball;
#select * from game limit 10;

drop table if exists batting_avg_wDATEDIFF ;
CREATE table batting_avg_wDATEDIFF (
batter int unsigned not null,
hit int unsigned not null,
atbat int unsigned not null,
local_date datetime not null,
diff_day int not null,
game_id int unsigned not null
) engine=InnoDB default charset latin1;

insert into batting_avg_wDATEDIFF (
game_id,batter,hit,atbat,local_date,diff_day)
select g.game_id,bc.batter,bc.hit,bc.atBat,g.local_date,DATEDIFF('2012-07-05 00:00:00' ,g.local_date) as date_diff
from game g
join batter_counts bc on
	bc.game_id = g.game_id ;
    
# select * from batting_avg_wDATEDIFF bawd;

set @Enddate = '2008-08-31';

drop table if exists rolling;

CREATE TABLE rolling SELECT a.batter,
	a.game_id,
    a.local_date AS End_date,
    b.local_date AS Beg_date,
    b.hit AS HIT,
    b.atbat AS ATBAT,
	DATEDIFF(a.local_date,b.local_date) as diff
    FROM
    batting_avg_wDATEDIFF a
         JOIN
    batting_avg_wDATEDIFF b ON a.batter = b.batter
    where DATEDIFF(a.local_date,b.local_date) between 0 and 100 and a.game_id= 12560;

select batter,sum(HIT)/sum(ATBAT) as AVG from rolling group by batter order by batter, sum(HIT)/sum(ATBAT);

# chmod a+x SQLBASH.sh
