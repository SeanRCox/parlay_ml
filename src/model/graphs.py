import dill as pickle
import torch
from torch_geometric.data import Data
from model.scraper import Game

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Game':
            return Game
        return super().find_class(module, name)

def get_data():
    data = []

    print("Formatting data for training...")

    list_of_games = []
    with open('data/new_games.pkl', 'rb') as file:
        while True:
            try:
                game = CustomUnpickler(file).load()
            except EOFError:
                break
            else:
                list_of_games.append(game)

    all_player_id_list = []  # List of all player IDs
    all_node_list = []
    for game in list_of_games:

        # Get player IDs and Stats
        player_ids = game.players
        for pid in player_ids:
            if pid not in all_player_id_list:
                all_player_id_list.append(pid)
                all_node_list.extend([pid*3, pid*3+1, pid*3+2])  # Adding Player nodes for pts, rbs, ast

        node_ids = []
        for pid in player_ids:
            node_ids.extend([pid*3, pid*3+1, pid*3+2])

        stat_nodes = []
        for stats in game.stats:
            for stat in stats:
                stat_nodes.append(stat)

        # Fully connected graph, make the edge_index have every node connected with every other
        edge_index = torch.tensor([[i, j] for i in range(len(stat_nodes)) \
                                for j in range(len(stat_nodes)) if i != j], dtype=torch.long).t().contiguous()
        
        # Predicting player stats from their player IDs and connections to each other player
        d = Data(edge_index=edge_index, y=stat_nodes, batch=None)
        d.node_ids = node_ids
        
        data.append(d)

    return data, all_node_list

def get_node_values(node_ids):
    node_values = []

    with open('data/player_data.pkl', 'rb') as file:
        player_dict = pickle.load(file)

    for node in node_ids[::3]:
        player = player_dict[node/3]
        node.values.extend([f"{player} Points", f"{player} Rebounds", f"{player} Assists"])

    return node_values

        
