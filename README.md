transform to xBPP to xBP/100P
create hdf5 file
evaluate player by season performance
evaluate teams by season performance

pitches : number of occurences
player : player_name
season : game_year
teams : hard to identify

#TODO (make Issues)
* establish evaluation suite
  * correlation of XBPP with actual BPP
	* RMSE on every pitch
  * RMSE on seasonal (xBPP * Pitches vs Bases)
* best vs worst pitch for 2019
* try to take pitching type into account
* use GANs to fill missing data



#plotting
statsbomb plot: https://i1.wp.com/statsbomb.com/wp-content/uploads/2020/02/Roberto-Firmino-Premier-League-2019_2020.png?resize=1024%2C774&ssl=1

checkout pybaseball

#use cases for xBPP
* Decision aid to when hook Pitchers
* Training Direction
* Expose Batter weaknesses (e.g., a "bad" curve ball might be better than a "good" fastball etc.)
* xBPP vs Contact quality (how many bases for that contact) -> good contact, lucky contact? Are some batter significantly converting low xBPP pitches into higher xBPP contacts
* are there batters who tire pitchers by getting most pitches per PA

# Tuning Options
* take pitch type into account
* use GANs to fill missing data

