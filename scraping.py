import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import csv

years = list(range(2005, 2023))
url_start = "https://www.sports-reference.com/cbb/schools/duke/{}-schedule.html"

#parsing the yearly data
all_games_from_2005_to_2022 = {}
games_num = 0
for year in years:
    url = url_start.format(year)
    data = requests.get(url)
    all_games_in_year = []
    page = data.text
    ########## extracting data ##########
    soup = BeautifulSoup(page, "html.parser")
    schedule = soup.find(id="schedule")
    # tr: specify different games
    all_the_tr = schedule.tbody.find_all('tr')
    #different games
    for game in all_the_tr:
        game_page = game.find('td', {'data-stat': 'date_game'})
        game_page1 = game_page.find('a') if game_page != None else None
        game_page2 = game_page1.get('href') if game_page1 != None else None
        #store all games for this year
        all_games_in_year.append(game_page2) if game_page2 != None else 0
        
    #store the list to the dict
    all_games_from_2005_to_2022[str(year)] = all_games_in_year
    games_num += len(all_games_in_year)

print(games_num)

# Practically extract the table for every game
game_url = "https://www.sports-reference.com/{}"
all_box_score_from_2005_to_2022 = []
for year in tqdm(years):
    box_score_list = []
    for date in tqdm(all_games_from_2005_to_2022[str(year)]):
        real_url = game_url.format(date)
        data = requests.get(real_url)
        page = data.text
        soup1 = BeautifulSoup(page, "html.parser")
        Duke_table = soup1.find(id="box-score-basic-duke")

        #I can also parse the "box-score-advanced-duke"s
        Duke_box_score = pd.read_html(str(Duke_table))[0]
    
        #Practically extract the table of the every game for both team scores
        line_score = soup1.find_all('div', {'class': 'score'} )
        line_score_pair = [line_score[0].text, line_score[1].text]

        #Data clean
        Duke_box_score = Duke_box_score.drop([5])
        Duke_box_score.iloc[-1,-1] = int(Duke_box_score.iloc[-1, -1])*2 - int(line_score_pair[0]) - int(line_score_pair[1])
        box_score_list.append(Duke_box_score)
        
        all_box_score_from_2005_to_2022.append(Duke_box_score)

    box_score_list = pd.concat(box_score_list)
    # box_score_list.to_csv('box{}'.format(year), index = False)
    print(Duke_box_score)

all_box_score_from_2005_to_2022 = pd.concat(all_box_score_from_2005_to_2022)
all_box_score_from_2005_to_2022.to_csv('box2005to2022', index = False)




    


