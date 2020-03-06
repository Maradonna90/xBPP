import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    data = pd.read_csv('xBPP.csv')
    p_data = data.groupby(['player_name', 'game_year'], sort=False).mean()
    p_data.loc[:, 'pitches'] = data.groupby(['player_name', 'game_year'], sort=False).count().loc[:, 'bases']
    p_data = p_data.rename({'bases': 'BPP'}, axis=1)
    p_data.loc[:, 'diff'] = p_data.loc[:, 'BPP'] - p_data.loc[:, 'xBPP']
    print(p_data)
    #print(data.groupby(['player_name', 'game_year'], sort=False).count())
    #TODO: LOWEST xBPP per season for all players. filter to low amount of pitches
    season = 2019.0
    min_pitches = 400
    s_data = p_data.loc[(p_data.index.get_level_values('game_year') == season)].copy()
    s_data = s_data[s_data['pitches'] >= min_pitches]
    top_xbpp = s_data.sort_values(by=['xBPP'])
    print(top_xbpp)
    flop_xbpp = s_data.sort_values(by=['xBPP'], ascending=False)
    print(flop_xbpp)
    top_diff = s_data.sort_values(by=['diff'])
    print(top_diff)
    flop_diff = s_data.sort_values(by=['diff'], ascending=False)
    print(flop_diff)
    s_data.loc[:, 'xbpp_pct'] = s_data.loc[:, 'xBPP'].rank(pct=True, ascending=False)
    print(s_data)
    players = ["Jordan Zimmermann", "Austin Adams", "Chris Bassitt"]
    pct_plot(players[::-1], s_data, str(int(season)))
def pct_plot(players, s_data, title):
    current_palette = sns.color_palette('colorblind')
    dark_palette = sns.color_palette('deep')
    #ax = sns.kdeplot(s_data['xBPP'], shade=True, color='gray', legend=False, linewidth=0.2)
    for i, pitcher in enumerate(players):
        ax = sns.kdeplot(s_data['xBPP'], shade=True, color='gray', legend=False, linewidth=0.0)
        pitcher_pct = s_data.loc[(s_data.index.get_level_values('player_name') == pitcher)].copy()
        pitcher_xbpp = pitcher_pct['xBPP'].values[0]
        pitcher_xbpp_pct = pitcher_pct['xbpp_pct'].values[0]
        print(pitcher_xbpp, pitcher_xbpp_pct)
        line = ax.get_lines()[-1]
        x, y = line.get_data()
        mask = x >= pitcher_xbpp
        x_m, y_m = x[mask], y[mask]
        plt.fill_between(x_m, y1=y_m, alpha=0.5, facecolor=current_palette[i])
        marker_offset = 0.00025
        ax.plot(pitcher_xbpp+marker_offset, -1.5, marker='^', color=dark_palette[i])
        plt.text(min(x), 10+i*10, "{:.3f}\nP {:.0f}".format(pitcher_xbpp, pitcher_xbpp_pct*100), color=current_palette[i], weight='bold')
    #SHIFT AFTER
    ax.set_xlim(ax.get_xlim()[::-1])
    #remove stuff
    ax.set_yticks((1, 6))
    ax.set_yticklabels(("{:.3f}".format(max(x)), 'xBPP'), fontweight='bold')
    ax.tick_params(axis='y', direction='in', pad=0, length=0, width=0, labelcolor='gray')
    ax2 = ax.twinx()
    ax2.tick_params(axis='y', direction='in', pad=0, length=0, width=0, labelcolor='gray')
    ax2.set_yticks([0.1])
    ax2.set_yticklabels("{:.3f}".format(min(x)), fontweight='bold')
    ax.set_xticklabels([])
    ax.set_xticks([])
    sns.despine(bottom=True, left=True)
    plt.tight_layout()
    plt.savefig("xbpp-dist-"+title+".pdf")
    plt.clf()

if __name__ == "__main__":
    main()
