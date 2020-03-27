import pandas as pd
import numpy as np
def main():
    filter_cols = ["release_speed", "release_pos_x", "release_pos_z", "pfx_x", "pfx_z", "plate_x", "plate_z", "release_spin_rate", "game_year", "player_name", "type", "events"]

    #TODO: run?
    filter_events = ["batter_interference",
                    "caught_stealing_2b",
                    "caught_stealing_3b",
                    "caught_stealing_home",
                    "other_out",
                    "pickoff_1b",
                      "pickoff_2b",
                      "pickoff_3b",
                      "pickoff_caught_stealing_2b",
                      "pickoff_caught_stealing_3b",
                      "pickoff_caught_stealing_home",
                      "catcher_interf",
                    "field_error",
                    "run"]
    #TODO: if events null use type_conv
    type_conv = {"B" : 0.25, "S" : 0}
    events_conv = {"field_out" : 0,
                "home_run" : 4,
                "single" : 1,
                "force_out" : 0,
                "double" : 2,
                "triple" : 3,
                "sac_fly": 0,
                "sac_bunt" : 0,
                "fielders_choice_out" : 0,
                "hit_by_pitch" : 1,
                "grounded_into_double_play" : 0,
                   "fielders_choice" : 0,
                   "double_play" : 0,
                   "sac_fly_double_play" : 0,
                   "triple_play" : 0,
                   "intent_walk" : 0.25,
                   "strikeout" : 0,
                   "strikeout_double_play" : 0,
                   "walk" : 0.25,
                   "sac_bunt_double_play" : 0,
                   "null" : -1
                  }
    chunksize = 10 ** 6
    cols = pd.read_csv("pitches.csv", nrows=1).columns
    #print(cols)
    for chunk in pd.read_csv("pitches.csv", chunksize=chunksize):
        #data = data.drop([4943598])
        chunk.columns = cols
        process_data(chunk, events_conv, filter_events, filter_cols)
def process_data(data, events_conv, filter_events, filter_cols):
    print(data.shape)
    data = data.drop(data[data['events'].isin(filter_events)].index)
    #print(data.shape)
    #data = data['events'].map({np.nan: 'null'})
    data['events'] = data['events'].fillna('null')
    data['bases'] = -1
    data['bases'] = data['events'].map(events_conv)
    data.loc[(data['events'] == 'null') & (data['type'] == 'B') ,'bases'] = 0.25
    data.loc[(data['events'] == 'null') & (data['type'] == 'S') ,'bases'] = 0
    #print(data.isna().sum())
    #print(data.shape)
    data = data.drop(['events', 'type'], axis=1)
    #data = data.dropna()
    print(data.shape)
    data[filter_cols[:-3]] = data[filter_cols[:-3]].apply(pd.to_numeric)
    data.to_hdf('pitches.h5', key='data', mode='a')
    #print(data.describe())

if __name__ == "__main__":
    main()
