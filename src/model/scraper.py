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
        
        
def get_games():
    game_file = open('data/game_data.pkl', 'ab')
    player_file = open('data/player_id_nums.pkl', 'wb')
        
    seasons = [i for i in range(2019, 2025)]  # Using last 5 seasons (2019-2024)
    
    all_player_dict = {}

    for year in seasons:
        games = scraper.season_schedule(season_end_year=year)  # Get all games in the given year

        game_dates = []  # List of tuples for days where game(s) were played
        for game in games:
            game_day = (game["start_time"].day, game["start_time"].month)

            if game_day not in game_dates:
                game_dates.append(game_day)

        for game_day in game_dates:
            success = False

            day = game_day[0]
            month = game_day[1]

            print(f"Getting stats for {month}/{day}/{year}")

            while not success:
                try:
                    # For some reason the scraper is erronieously including days where there were no games? Double check to make sure
                    box_scores = scraper.player_box_scores(day=day, month=month, year=year)
                    player_data = pd.DataFrame.from_dict(box_scores)  # Get the player data as a pandas dataframe

                    for col in player_data.columns:
                        # Locations/Teams are stored as enums for whatever reason, so just get their values
                        if isinstance(player_data[col].iloc[0], Team) or isinstance(player_data[col].iloc[0], Location):
                            player_data[col] = player_data[col].apply(lambda x: x.value)

                    sorted_data = player_data.sort_values('team')  # Sort data on teams so players are grouped together
                    # Filter out unnecessary columns
                    cleaned_data = sorted_data.filter(['name', 'team', 'opponent', 'location', 'points', 'rebounds', 'assists', 'seconds_played'])

                    # Group teams together
                    grouped_data = cleaned_data.groupby('team')
                    # Split teams in seperate dataframes
                    team_data = [group for _, group in grouped_data]

                    game_scores = []
                    for team in team_data:
                        # Group teams with their opponents
                        if team["location"].iloc[0] == "HOME":
                            opponent = team["opponent"].iloc[0]
                            for other_team in team_data:
                                if other_team["team"].iloc[0] == opponent:
                                    # Concatenate two opposing teams into a single DF
                                    g = pd.concat([team, other_team])
                                    g = g.nlargest(n=16, columns=['seconds_played'])  # Get top 16 players by minutes played
                                    g = g.drop(columns=['team', 'opponent', 'seconds_played', 'location'])  # Don't need these columns anyomore
                                    game_scores.append(g)


                    for box_score in game_scores:
                        
                        player_id_list = []
                        player_stats_matrix = []
                        for _, row in box_score.iterrows():
                            # Iterate through the box score
                            if row['name'] not in all_player_dict.keys():
                                all_player_dict[row['name']] = len(all_player_dict.keys())+1
                                pickle.dump(all_player_dict, player_file) # update all player dict

                            # Add players ID and stats
                            player_id_list.append(all_player_dict[row['name']])
                            player_stats_matrix.append([row['points'], row['rebounds'], row['assists']])

                        # Make a new game object and add it to the list
                        pickle.dump((Game(player_id_list, player_stats_matrix)), game_file)

                    success = True

                except KeyError as e:
                    break
                except Exception as e:
                    print(f"Error on {month}/{day}/{year}: {e}")
                    print("Sleeping...")
                    time.sleep(60)

def main(): 
    get_games()

if __name__=="__main__": 
    main() 
