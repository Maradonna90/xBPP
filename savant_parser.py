import requests
from bs4 import BeautifulSoup
import csv
from io import StringIO
f = open("savant.html")
soup = BeautifulSoup(f, 'html.parser')
f.close()

csv_url = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea=2019%7C2018%7C2017%7C2016%7C2015%7C2014%7C2013%7C2012%7C2011%7C2010%7C2009%7C2008%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&hfFlag=&hfPull=&pitchers_lookup%5B%5D=[pitcher_id]&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&min_pas=0&type=details&"

select_id = "pitchers_lookup"

options = soup.find(id=select_id).children
pitcher_ids = []
for opt in options:
    try:
        pitcher_ids.append(opt['value'])
    except:
       pass
pitcher_ids = pitcher_ids[2296:]
for j, pitcher in enumerate(pitcher_ids):
    print("{0}/{1} | {2}% {3}".format(j, len(pitcher_ids), round(j/len(pitcher_ids)*100, 2), pitcher))
    p_url = csv_url.replace("[pitcher_id]", pitcher)
    r = requests.get(p_url)
    f = StringIO(r.text)
    reader = csv.reader(f, delimiter=',')
    with open('pitches.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        for i, row in enumerate(reader):
            if (j == 0 and i == 0) or (i != 0):
                writer.writerow(row)
