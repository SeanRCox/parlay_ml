import pickle
import math
import numpy as np
from scipy.stats import norm

def get_stats_represented_by_nodes(node_ids):
    node_values = []

    with open('data/player_id_nums.pkl', 'rb') as file:
        player_dict = pickle.load(file)

    for node in node_ids[::3]:
        for player, player_id in player_dict.items():
            if player_id == int(node/3):
                node_values.extend([f"{player} Points", f"{player} Rebounds", f"{player} Assists"])

    return node_values

def get_player_from_node(node_id):
    with open('data/player_id_nums.pkl', 'rb') as file:
        player_dict = pickle.load(file)

    print(player_dict)
    target_player_id = math.floor(node_id/3)
    for player, player_id  in player_dict.items():
        if player_id == target_player_id:
            return player

def get_lines():
    pass

def make_predictions(mu, var, pi, nodes, lines):
    """
    makes an intiial prediction of the best bet based on the predicted GMMs and the betting lines
    """
    mu, var, pi = mu.to_list(), var.to_list(), pi.to_list()

    x = np.linspace(mu.min() - 3*np.sqrt(var.max()), 
                        mu.max() + 3*np.sqrt(var.max()), 1000)

    overs_likelihood = []
    unders_likelihood = []

    for i in range(len(nodes)):
        gmm = np.zeros_like(x)
        for j in range(0,5):
            component = pi[i][j] * norm.pdf(x, mu[i][j], np.sqrt(var[i][j]))
            gmm += component

        cdf_value = np.sum(gmm[x > lines[i]])
        overs_likelihood.append(1.0 - cdf_value)
        unders_likelihood.append(cdf_value)

    prediction_dict = {}
    stats_to_predict = get_stats_represented_by_nodes(nodes)

    for stat, over_likelihood, under_likelihood \
        in zip(stats_to_predict, overs_likelihood, unders_likelihood):
        
        over_or_under = 'Over' if over_likelihood > 0.5 else 'Under'
        confidence = over_likelihood if over_likelihood > 0.5 else under_likelihood
        prediction_dict[stat] = (over_or_under, confidence)

    highest_confidence = 0
    most_likely_bet = ('', '')
    for stat, prediction in prediction_dict.items:
        if prediction[1] > highest_confidence:
            highest_confidence = prediction[1]
            most_likely_bet = (stat, prediction[0])

    return most_likely_bet

print(get_player_from_node(56))