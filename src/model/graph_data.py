import dill as pickle
import torch
from torch_geometric.data import Data
from model.get_data import Game

class Unpickle_Games(pickle.Unpickler):
    """
    Used to unpickle game data from games_data.pkl
    """
    def find_class(self, module, name):
        if name == 'Game':
            return Game
        return super().find_class(module, name)

def graphify_data():
    data = []

    # Loop through file and get all games
    games = []
    with open('data/new_games.pkl', 'rb') as file:
        while True:
            try:
                game = Unpickle_Games(file).load()
            except EOFError:
                break
            else:
                games.append(game)

    possible_nodes = []

    for game in games:

        node_ids = []  # Each player has 3 nodes associated with them, 1 for pts, rbs, and asts
        # Each players nodes are the floor of the node ID divided by 3. Ex: Player 17 has nodes 51, 52, 53
        for player_id in game.players:
            pts_node, rbs_node, ast_node = player_id*3, player_id*3+1, player_id*3+2 
            node_ids.extend([pts_node, rbs_node, ast_node])  # Adding Player nodes for pts, rbs, ast

            # Keep list of all possible nodes
            if pts_node not in possible_nodes:
                possible_nodes.extend([pts_node, rbs_node, ast_node])

        # These are the result nodes
        result_node_values = []
        for stats in game.stats:
            result_node_values.extend([stats])

        assert len(node_ids) == len(result_node_values)  # Input nodes and output nodes should have same length

        # Fully connected graph, make the edge_index have every node connected with every other
        edge_index = torch.tensor([[i, j] for i in range(len(node_ids)) \
                                for j in range(len(node_ids)) if i != j], dtype=torch.long).t().contiguous()
        
        # Predicting player stats from their player IDs and connections to each other player
        d = Data(edge_index=edge_index, y=result_node_values, batch=None)
        d.node_ids = node_ids
        
        data.append(d)

    return data, possible_nodes

        
