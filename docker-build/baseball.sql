use baseball;
#select * from game limit 10;

create table if not exists batting_avg_wDATEDIFF ;

insert into batting_avg_wDATEDIFF (
 game_id,batter,hit,atbat,local_date,diff_day)
select g.game_id,bc.batter,bc.hit,bc.atBat,g.local_date,DATEDIFF('2012-07-05 00:00:00' ,g.local_date) as date_diff
from game g
join batter_counts bc on
	bc.game_id = g.game_id ;
    
create table if not exists rolling;

insert into rolling SELECT 
   a.batter,
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

