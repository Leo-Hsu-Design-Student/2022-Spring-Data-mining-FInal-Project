import csv
import pandas as pd

def NTU_data():
    box_2017_19 = pd.read_csv('NTU_box_score.csv')
    # print(box_2017_19)
    total_per_game = box_2017_19.loc[box_2017_19['球員'] == '團隊'].iloc[:, 1:]

    x = total_per_game.iloc[:, 1:-1]
    x.columns = ['FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    y = total_per_game.iloc[:, 0].to_numpy().astype(int) - total_per_game.iloc[:, -1].to_numpy().astype(int)
    y_dummy = y
    y_dummy[y_dummy>0] = 1
    y_dummy[y_dummy<0] = 0
    return x, y, y_dummy
# NTU_data()