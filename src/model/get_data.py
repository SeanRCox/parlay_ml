"""
Get and format all Pts, Rbs, Ast for each player for each game
"""
import time
import pickle
import pandas as pd

from basketball_reference_web_scraper.basketball_reference_web_scraper import client as scraper
from basketball_reference_web_scraper.basketball_reference_web_scraper.data import Team, Location

class Game:
    """
    Make a game class for handling the player list and their stats
    """
    def __init__(self, players_dict, stats_list):
        self.players = players_dict
        self.stats = stats_list


def get_game_data():
    game_file = open('data/game_data.pkl', 'ab')
    player_file = open('data/player_id_nums.pkl', 'wb')
        
    seasons = [i for i in range(2019, 2025)]  # Using last 5 seasons (2019-2024)

    games = []
    for year in seasons:
        games.extend([scraper.season_schedule(season_end_year=year)])

    days_with_games = []
    for game in games:
        game_day = (game["start_time"].month, game["start_time"].day, game["start_time"].year)
        if game_day not in days_with_games:
                days_with_games.append(game_day)


    player_dict = {}  # For mapping players to their player ids. [name] = player_id

    for day in days_with_games:
        m, d, y = day[0], day[1], day[2]

        print(f"Getting stats for games played on {m}/{d}/{y}...")

        rate_limited = False

        while not rate_limited:
            try:
                all_box_scores = scraper.player_box_scores(day=d, month=m, year=y)  # Get all box scores for the given day
                player_data = pd.DataFrame.from_dict(all_box_scores)  # Get the player data as a pandas dataframe

                # Locations/Teams are stored as enums for whatever reason, so just get their values
                for col in player_data.columns:
                    if isinstance(player_data[col].iloc[0], Team) or isinstance(player_data[col].iloc[0], Location):
                        player_data[col] = player_data[col].apply(lambda x: x.value)

                sorted_data = player_data.sort_values('team')  # Sort data on teams so players on the same team are grouped together
                
                # Filter out unnecessary columns
                cleaned_data = sorted_data.filter(['name', 'team', 'opponent', 'location', 'points', 'rebounds', 'assists', 'seconds_played'])

                grouped_data = cleaned_data.groupby('team')  # Group teams together
                team_data = [group for _, group in grouped_data]  # Split teams in seperate dataframes

                # Group teams with their opponents to make a single DF for each game
                game_box_scores = []
                for team in team_data:
                    if team["location"].iloc[0] == "HOME":
                        opponent = team["opponent"].iloc[0]
                        for other_team in team_data:
                            if other_team["team"].iloc[0] == opponent:
                                # Concatenate two opposing teams into a single DF
                                g = pd.concat([team, other_team])
                                g = g.nlargest(n=16, columns=['seconds_played'])  # Get top 16 players by minutes played
                                g = g.drop(columns=['team', 'opponent', 'seconds_played', 'location'])  # Don't need these columns anyomore
                                game_box_scores.append(g)

                for box_score in game_box_scores:
                    player_id_list = []
                    player_stats_matrix = []

                    # Iterate through the box score
                    for _, row in box_score.iterrows():
                        if row['name'] not in player_dict.keys():
                            player_dict[row['name']] = len(player_dict.keys())+1
                            pickle.dump(player_dict, player_file) # update all player dict

                        player_id_list.append(player_dict[row['name']])  # Add a players ID to the ID list
                        player_stats_matrix.append([row['points'], row['rebounds'], row['assists']])

                    # Make a new game object and add it to the list
                    pickle.dump((Game(player_id_list, player_stats_matrix)), game_file)

            except KeyError as e:
                print("No Games on this day...")
                break
            except Exception as e:
                print("Hit rate limit, waiting...")
                time.sleep(60)

def main(): 
    get_game_data()

if __name__=="__main__": 
    main() 
