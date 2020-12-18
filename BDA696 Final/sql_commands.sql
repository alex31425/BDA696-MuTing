use baseball;


# drop table if exists team_pitching_count_c;

create table if not exists team_pitching_count_c
select tbc.*,pc.startingPitcher,sum(pc.startingInning) as startingInning ,sum(pc.endingInning) as endingInning
from team_pitching_counts tbc
join pitcher_counts pc on pc.game_id = tbc.game_id and pc.team_id = tbc.team_id
group by pc.team_id,pc.game_id
;



ALTER TABLE baseball.team_pitching_count_c 
CHANGE COLUMN finalScore P_finalScore INT NULL DEFAULT 0 ,
CHANGE COLUMN win P_win INT NULL DEFAULT 0 ,
CHANGE COLUMN plateApperance P_plateApperance INT NULL DEFAULT 0 ,
CHANGE COLUMN atBat P_atBat INT NULL DEFAULT 0 ,
CHANGE COLUMN Hit P_Hit INT NULL DEFAULT 0 ,
CHANGE COLUMN bullpenOutsPlayed P_bullpenOutsPlayed INT NULL DEFAULT 0 ,
CHANGE COLUMN bullpenWalk P_bullpenWalk INT NULL DEFAULT 0 ,
CHANGE COLUMN bullpenIntentWalk P_bullpenIntentWalk INT NULL DEFAULT 0 ,
CHANGE COLUMN bullpenHit P_bullpenHit INT NULL DEFAULT 0 ,
CHANGE COLUMN caughtStealing2B P_caughtStealing2B INT NULL DEFAULT 0 ,
CHANGE COLUMN caughtStealing3B P_caughtStealing3B INT NULL DEFAULT 0 ,
CHANGE COLUMN caughtStealingHome P_caughtStealingHome INT NULL DEFAULT 0 ,
CHANGE COLUMN stolenBase2B P_stolenBase2B INT NULL DEFAULT 0 ,
CHANGE COLUMN stolenBase3B P_stolenBase3B INT NULL DEFAULT 0 ,
CHANGE COLUMN stolenBaseHome P_stolenBaseHome INT NULL DEFAULT 0 ,
CHANGE COLUMN toBase P_toBase INT NULL DEFAULT 0 ,
CHANGE COLUMN updatedDate P_updatedDate TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP ,
CHANGE COLUMN Batter_Interference P_Batter_Interference FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Bunt_Ground_Out P_Bunt_Ground_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Bunt_Groundout P_Bunt_Groundout FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Bunt_Pop_Out P_Bunt_Pop_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Catcher_Interference P_Catcher_Interference FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN `Double` P_Double FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Double_Play P_Double_Play FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Fan_interference P_Fan_interference FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Field_Error P_Field_Error FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Fielders_Choice P_Fielders_Choice FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Fielders_Choice_Out P_Fielders_Choice_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Fly_Out P_Fly_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Flyout P_Flyout FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Force_Out P_Force_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Forceout P_Forceout FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Ground_Out P_Ground_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Grounded_Into_DP P_Grounded_Into_DP FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Groundout P_Groundout FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Hit_By_Pitch P_Hit_By_Pitch FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Home_Run P_Home_Run FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Intent_Walk P_Intent_Walk FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Line_Out P_Line_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Lineout P_Lineout FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Pop_Out P_Pop_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Runner_Out P_Runner_Out FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Sac_Bunt P_Sac_Bunt FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Sac_Fly P_Sac_Fly FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Sac_Fly_DP P_Sac_Fly_DP FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Sacrifice_Bunt_DP P_Sacrifice_Bunt_DP FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Single P_Single FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Strikeout P_Strikeout FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN `Strikeout_-_DP` `P_Strikeout_-_DP` FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN `Strikeout_-_TP` `P_Strikeout_-_TP` FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Triple P_Triple FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Triple_Play P_Triple_Play FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN Walk P_Walk FLOAT NULL DEFAULT 0 ,
CHANGE COLUMN startingPitcher P_startingPitcher INT NULL DEFAULT 0 ,
CHANGE COLUMN startingInning P_startingInning DOUBLE NULL DEFAULT 0 ,
CHANGE COLUMN endingInning P_endingInning DOUBLE NULL DEFAULT 0 
;



# drop table if exists team_count_all;

create table if not exists team_count_all
select tbc.*,b.away_runs,tr.home_streak,tr.away_streak,tpcc.P_finalScore,tpcc.P_win,tpcc.P_plateApperance,tpcc.P_atBat,tpcc.P_Hit,
tpcc.P_bullpenOutsPlayed,tpcc.P_bullpenWalk,tpcc.P_bullpenIntentWalk,tpcc.P_bullpenHit,tpcc.P_caughtStealing2B,tpcc.P_caughtStealing3B,
tpcc.P_caughtStealingHome,tpcc.P_stolenBase2B,tpcc.P_stolenBase3B,tpcc.P_stolenBaseHome,tpcc.P_toBase,
tpcc.P_updatedDate,tpcc.P_Batter_Interference,tpcc.P_Bunt_Ground_Out,tpcc.P_Bunt_Groundout,tpcc.P_Bunt_Pop_Out,tpcc.P_Catcher_Interference,
tpcc.P_Double,tpcc.P_Double_Play,tpcc.P_Fan_interference,tpcc.P_Field_Error,tpcc.P_Fielders_Choice,tpcc.P_Fielders_Choice_Out,tpcc.P_Fly_Out,
tpcc.P_Flyout,tpcc.P_Force_Out,tpcc.P_Forceout,tpcc.P_Ground_Out,tpcc.P_Grounded_Into_DP,tpcc.P_Groundout,tpcc.P_Hit_By_Pitch,tpcc.P_Home_Run,tpcc.P_Intent_Walk,
tpcc.P_Line_Out,tpcc.P_Lineout,tpcc.P_Pop_Out,tpcc.P_Runner_Out,tpcc.P_Sac_Bunt,tpcc.P_Sac_Fly,tpcc.P_Sac_Fly_DP,
tpcc.P_Sacrifice_Bunt_DP,tpcc.P_Single,tpcc.P_Strikeout,tpcc.P_Triple,tpcc.P_Triple_Play,
tpcc.P_Walk,(tpcc.P_startingInning) as P_startingInning,(tpcc.P_endingInning) as P_endingInning
from team_batting_counts tbc 
join team_pitching_count_c tpcc on tbc.game_id = tpcc.game_id and tbc.team_id = tpcc.team_id
join team_results tr on tr.team_id = tbc.team_id and tr.game_id = tbc.game_id
join boxscore b on tbc.game_id = b.game_id group by team_id, game_id
;




# drop table if exists roll_200_day;

create table if not exists roll_200_day 
select
g1.local_date as g1_date,
g2.local_date as g2_date,
tbc1.team_id as team_id,
tbc1.game_id as game_id,
nullif(sum(tbc2.atBat),0) as AB,
nullif(sum(tbc2.Hit),0) as H,
nullif(sum(tbc2.Single),0) as B,
nullif(sum(tbc2.Double),0) as 2B,
nullif(sum(tbc2.Triple),0) as 3B,
nullif(sum(tbc2.Home_Run),0) as HR,
nullif(sum(tbc2.Strikeout),0) as K,
nullif(sum(tbc2.Sac_Fly),0) as SF,
nullif(sum(tbc2.Walk),0) as BB,
nullif(sum(tbc2.Ground_Out),0) as GB,
nullif(sum(tbc2.Fly_Out),0) as FB,
nullif(sum(tbc2.Hit_By_Pitch),0) as HBP,
nullif(sum(tbc2.Intent_Walk),0) as IBB,
nullif(sum(tbc2.plateApperance),0) as PA,
nullif(sum(tbc2.Single+2*tbc2.Double+3*tbc2.Triple+4*tbc2.Home_Run),0) as TB,
nullif(sum(tbc2.Hit+tbc2.Walk+tbc2.Hit_By_Pitch),0) as TOB,
nullif(sum(tbc2.Double_Play),0) as DP,
nullif(sum(tbc2.Field_Error),0) as E,
nullif(sum(tbc2.Sac_Bunt),0) as SB,
nullif(sum(tbc2.P_Walk),0) as P_BB,
nullif(sum(tbc2.P_endingInning-tbc2.P_startingInning),0) as P_IP,
nullif(sum(tbc2.P_Hit_by_Pitch),0) as P_HPB,
nullif(sum(0.89*(1.255*(tbc2.P_Hit-tbc2.P_Home_Run)+4*tbc2.P_Home_Run)+0.56*(tbc2.P_Walk+tbc2.P_Hit_by_Pitch-tbc2.P_Intent_Walk)),0) as P_PTB,
nullif(sum(tbc2.P_Intent_Walk),0) as P_IBB,
# sum(tbc2.P_career_bb) as P_BFP,
nullif(sum(tbc2.P_Ground_out),0) as P_GB,
nullif(sum(tbc2.P_Fly_Out),0) as P_FB,
nullif(sum(tbc2.P_Hit_By_Pitch),0) as P_HBP,
nullif(sum(tbc2.P_Home_Run),0) as P_HR,
nullif(avg(tbc2.P_Home_Run),0) as P_AvgHR,
nullif(sum(tbc2.P_Strikeout),0) as P_K,
nullif(sum(tbc2.P_Double_Play),0) as P_2B_O,
nullif(sum(tbc2.home_streak),0) as home_streak,
nullif(sum(tbc2.away_streak),0) as away_streak,
nullif(9*(avg(tbc2.away_runs)/(tbc2.P_endingInning-tbc2.P_startingInning)),0) as ERA
from team_count_all tbc1
join team t on tbc1.team_id = t.team_id
join game g1 on g1.game_id = tbc1.game_id and g1.type = "R"
join team_count_all tbc2 on tbc1.team_id = tbc2.team_id
join game g2 on g2.game_id = tbc2.game_id and g2.type = "R" and g2.local_date < g1.local_date 
and g2.local_date >= date_add(g1.local_date, interval - 200 day)
group by tbc1.team_id, tbc1.game_id, g1.local_date
order by tbc1.team_id,g1.local_date;





# drop table if exists data_final;

create table if not exists data_final
SELECT 
    g.local_date,
    g.game_id,
    g.home_team_id,
    g.away_team_id,
    case when b.away_runs < b.home_runs then 1
    when b.away_runs > b.home_runs then 0
    else 0 end as home_team_wins,
    (r2dh.H / r2dh.AB) / (r2da.H / r2da.AB) AS BA_diff,
    ((r2dh.H - r2dh.HR) / (r2dh.AB - r2dh.K - r2dh.HR + r2dh.SF)) / ((r2da.H - r2da.HR) / (r2da.AB - r2da.K - r2da.HR + r2da.SF)) AS BABIP_diff,
    (r2dh.BB / r2dh.K) / (r2da.BB / r2da.K) AS BBK_diff,
    (r2dh.GB / nullif((r2dh.FB),0)) / nullif((r2da.GB / nullif((r2da.FB),0)),0) AS GBFB_diff,
    r2dh.HBP / nullif(r2da.HBP,0) AS HBP_diff,
    r2dh.HR / nullif(r2da.HR,0) AS HR_diff,
    (nullif(r2dh.HR,0) / r2dh.H) / nullif((nullif(r2da.HR,0) / r2da.H),0) AS HRH_diff,
    r2dh.IBB / nullif(r2da.IBB,0) AS IBB_diff,
    ((r2dh.TB-r2dh.B)/r2dh.AB)/nullif(((r2da.TB-r2da.B)/r2da.AB),0) AS ISO_diff,
    ((r2dh.TB) / r2dh.AB) / ((r2da.TB) / r2da.AB) AS SLG_diff,
    ((r2dh.H + r2dh.BB + r2dh.HBP) / (r2dh.AB + r2dh.BB + r2dh.HBP + r2dh.SF)) / ((r2da.H + r2da.BB + r2da.HBP) / (r2da.AB + r2da.BB + r2da.HBP + r2da.SF)) AS OBP_diff,
    ((r2dh.AB * (r2dh.H + r2dh.BB + r2dh.HBP) + r2dh.TB * (r2dh.AB + r2dh.BB + r2dh.SF + r2dh.HBP)) / (r2dh.AB * (r2dh.AB + r2dh.BB + r2dh.HBP + r2dh.SF))) / ((r2da.AB * (r2da.H + r2da.BB + r2da.HBP) + r2da.TB * (r2da.AB + r2da.BB + r2da.SF + r2da.HBP)) / (r2da.AB * (r2da.AB + r2da.BB + r2da.HBP + r2da.SF))) AS OBS_diff,
    ((r2dh.PA) / r2dh.K) / ((r2da.PA) / r2da.K) AS PAK_diff,
    (((r2dh.H + r2dh.BB) * r2dh.TB) / (r2dh.AB + r2dh.BB)) / (((r2da.H + r2da.BB) * r2da.TB) / (r2da.AB + r2da.BB)) AS RC_diff,
    r2dh.SF / nullif(r2da.SF,0) AS SF_diff,
    r2dh.SB / nullif(r2da.SB,0) AS SB_diff,
    (r2dh.SF + r2dh.SB) / nullif((r2da.SF + r2da.SB),0) AS SFSB_diff,
    r2dh.TB / r2da.TB AS TB_diff,
    r2dh.TOB / r2da.TOB AS TOB_diff,
    r2dh.DP / nullif(r2da.DP,0) AS DP_diff,
    r2dh.E / nullif(r2da.E,0) AS E_diff,
    (9 * r2dh.P_BB / r2dh.P_IP) AS P_BBP_diff,
    ((3 + (13 * r2dh.P_HR + 3 * (r2dh.P_BB + r2dh.P_HBP) - 2 * r2dh.P_K)) / (nullif(r2dh.P_IP,0)) / nullif(((3 + (13 * r2da.P_HR + 3 * (r2da.P_BB + r2da.P_HBP) - 2 * r2da.P_K)) / r2da.P_IP),0)) AS P_DICE_diff,
    r2dh.ERA / nullif(r2da.ERA,0) AS P_ERA,
    (r2dh.P_GB / nullif((r2dh.P_FB),0)) / nullif((r2da.P_GB / nullif((r2da.P_FB),0)),0) AS P_GBFB_diff,
    r2dh.P_HPB / nullif(r2da.P_HPB,0) AS P_HBP_diff,
    r2dh.P_HR / nullif(r2da.P_HR,0) AS P_HR_diff,
    (r2dh.P_avgHR / r2dh.P_IP) / nullif((r2da.P_avgHR / r2da.P_IP),0) AS P_HR9_diff,
    r2dh.P_K / r2da.P_K AS P_K_diff,
    ((r2dh.P_K + r2dh.P_BB) / r2dh.P_IP) / ((r2da.P_K + r2da.P_BB) / r2da.P_IP) AS P_PFR_diff,
    ((r2dh.P_HR + r2dh.P_BB) / r2dh.P_IP) / ((r2da.P_HR + r2da.P_BB) / r2da.P_IP) AS P_WHIP_diff,
    r2dh.P_2B_O / nullif(r2da.P_2B_O,0) AS P_DP_diff,
    (r2dh.P_K / r2dh.P_IP) / (r2da.P_K / r2da.P_IP) AS P_SIP_diff,
    (r2dh.P_BB / r2dh.P_IP) / (r2da.P_BB / r2da.P_IP) AS P_BB_diff,
    (r2dh.P_K / r2dh.P_BB) / nullif((r2da.P_K / r2da.P_BB),0) AS P_SBB_diff   
FROM
    game g
        JOIN
    roll_200_day r2dh ON g.game_id = r2dh.game_id
        AND g.home_team_id = r2dh.team_id
        JOIN
    roll_200_day r2da ON g.game_id = r2da.game_id
        AND g.away_team_id = r2da.team_id
        join boxscore b on b.game_id = g.game_id;



