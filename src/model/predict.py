import pickle

def get_node_values(node_ids):
    node_values = []

    with open('data/player_id_nums.pkl', 'rb') as file:
        player_dict = pickle.load(file)

    for node in node_ids[::3]:
        print(node)
        for player, pid in player_dict.items():
            print(player, pid)
            if pid == int(node/3):
                node_values.extend([f"{player} Points", f"{player} Rebounds", f"{player} Assists"])

    return node_values

def get_lines():
    pass

def make_predictions(mu, var, pi, nodes, lines):
    pass

print(get_node_values((3,4,5,6,7,8,51,52,53)))