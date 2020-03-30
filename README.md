# Usage
1. Parser for getting the data
2. Preprocessing to create 'bases'
3. train_boosting to train models
4. predct_boosting to predict xBPP
5. seasonal_analysis for analysis

evaluate player by season performance

pitches : number of occurences
player : player_name
season : game_year
teams : hard to identify

# TODO (make Issues)
* best vs worst pitch for 2019
* try to take pitching type into account
* use GANs to fill missing data
* optimize weights for combined predictions

#use cases for xBPP
* Decision aid to when hook Pitchers
* Training Direction
* Expose Batter weaknesses (e.g., a "bad" curve ball might be better than a "good" fastball etc.)
* xBPP vs Contact quality (how many bases for that contact) -> good contact, lucky contact? Are some batter significantly converting low xBPP pitches into higher xBPP contacts
* are there batters who tire pitchers by getting most pitches per PA

# Tuning Options
* take pitch type into account
* use GANs to fill missing data

#other ideas
* P/PA for batters. To see how much they drain pitchers (follow up for SP/RP sustain curves)
* 
