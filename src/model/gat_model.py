import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch_geometric.nn import GCNConv, GATConv

from model.plot_distributions import plot_gmm
from model.graph_data import get_data

class GNN_MDN(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dimensions, num_mixtures):
        """
        num_nodes : the total number of players for all games
        embedding_dim : the dimension for the player embeddings
        hidden_dim : the dimension for the convolutional and linear layers
        num_mixtures : number of values for each variable of the mixture normal distribution

        embedding : player emeddings, Each player is a row, with emedding_dim columns, batch_size * emedding_dim
        conv_layer_1 :  first convolutional layer, using GCN function (A * H[l] * W[l]), output a new matrix of batch_size * hidden_dim
        conv_layer_2 :  second convolutional layer, batch_size * hidden_dim
        fully_connected_layer : linear, fully connected layer, batch_size * hidden_dim

        mu : num_stats amount of linear layers for mu, num_mixtures outputs
        var : num_stats amount of linear layers for var, num_mixtures outputs
        pi : num_stats amount of linear layers for pi, num_mixtures outputs
        """
        super(GNN_MDN, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)  # Embed Node IDS
        init.xavier_uniform_(self.embedding.weight)
        self.conv_layer_1 = GATConv(embedding_dim, hidden_dimensions, heads=8, concat=True, dropout=0.6)  # First convolutional layer
        self.conv_layer_2 = GATConv(hidden_dimensions * 8, hidden_dimensions, heads=1, concat=True, dropout=0.6)  # Second convolutional layer
        self.dropout = nn.Dropout(0.6)
        self.fully_connected_layer = nn.Linear(hidden_dimensions, hidden_dimensions)  # Linear fully connected layer

        self.num_mixtures = num_mixtures  # Number of mixtures of the GMM Distribution, higher will capture more nuance of distribution
        self.mu = nn.Linear(hidden_dimensions, self.num_mixtures)
        self.var = nn.Linear(hidden_dimensions, self.num_mixtures)
        self.pi = nn.Linear(hidden_dimensions, self.num_mixtures)

    def forward(self, node_ids, edge_index):
        """
        Forward pass through the network

        player_ids : the unique ids of the players for this specific game
        edge_index : the connections between nodes (players) on the graph, always a fully connected graph
        
        """
        node_ids = torch.tensor(node_ids, dtype=torch.long)
        node_embeddings = self.embedding(node_ids)  # Get the embeddings for the nodes in this game
        x = node_embeddings  # Use the embeddings as input. Could also include game-specific info like location
        x = self.conv_layer_1(x, edge_index)  # First convolotional layer
        x = F.elu(x)
        print(f"X after first conv layer {x}")
        x = self.conv_layer_2(x, edge_index)  # Second convoluotional layer
        x = F.elu(x)
        print(f"X after second conv layer {x}")
        x = self.dropout(x)
        x = self.fully_connected_layer(x)  # Fully connected linear layer
        x = F.elu(x)
        print(f"X after fully connected layer {x}")

        # Last layer for predicting variables to our Mixture dist, then create tensors of batch_size * num_mixutres for each variable
        mu = self.mu(x)
        var = F.softplus(self.var(x))

        pi = F.softmax(self.pi(x), dim=1)

        return mu, var, pi
    
def loss(mu, var, pi, y):
    """
    Loss function using negative-log likelihood over our mixture normal distributions.

    mu: means of the distribtuions
    var: variances of the distributions
    pi: percent of each distrubtion contributing to the mixture
    
    """
    num_nodes, num_mixtures = mu.shape  # Extract the dimensions of our input tensors for later use

    y = torch.tensor(y, dtype=torch.float)
    y = y.unsqueeze(1)  # Reshape stats list into a 3D tensor 
    
    """
    """

    y = y.expand(num_nodes, num_mixtures)  # We need to reshape our new y tensor to be in the same format as our input tensor

    """
    """

    # Log Probability Computation

    
    # Define the normal distribution
    N = torch.distributions.Normal(mu, torch.sqrt(var))

    log_probability = N.log_prob(y)
    #print(y)
    #print(log_probability)

    weighted_log_probability = log_probability + torch.log(pi)

    sum_of_log_probabilites = torch.logsumexp(weighted_log_probability, dim=1)

    total_log_likelihood = torch.sum(sum_of_log_probabilites)

    negative_log_likelihood = -total_log_likelihood

    entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=1).mean()

    entropy_weight = 0.1
    total_loss = negative_log_likelihood - entropy_weight * entropy

    return -total_log_likelihood

def train(model, data_loader, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        # set model to training mode
        model.train()

        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad() # Reset gradients

            node_ids = data.node_ids # Get player ids for given game (input)
            edge_index = data.edge_index  # Get edge index for given game (connections between players)
            y = data.y  # Target data: players stats for given game 

            # Get the parameters for the MDN from the model given the player ids
            node_ids = torch.tensor(node_ids, dtype=torch.long)
            mu, var, pi = model(node_ids, edge_index)

            
            print(f"Node IDs: {node_ids[:5]}")
            print(f"Embeddings: {model.embedding(node_ids[:5])}")
            print(f"Mu: {mu[:5]}")
            print(f"Var: {var[:5]}")
            print(f"Pi: {pi[:5]}")
            

            # Compute loss 
            l = loss(mu, var, pi, y) 

            l.backward()  # Compute gradient of loss 
            optimizer.step()  # Update the model based on graident

            total_loss = l.item()  # Add up total loss for averaging at the end

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def main(): 
    data, node_list = get_data()

    model = GNN_MDN(num_nodes=len(node_list), embedding_dim=48, hidden_dimensions=64, num_mixtures=5)
    
    train(model, data[:-1], 10, 0.001)

    print(data[-1].node_ids)
    node_ids = data[-1].node_ids
    node_ids = torch.tensor(node_ids, dtype=torch.long)
    mu, var, pi = model(node_ids, data[-1].edge_index)
    print(f"Embeddings: {model.embedding(node_ids[:5])}")
    print("Predicted means (mu):", mu)
    print("Predicted variances (var):", var)
    print("Predicted mixture weights (pi):", pi)
    print("True value (y):", data[-1].y)

    for i in range(48):
        plot_gmm(mu, var, pi, data[-1].y, i, f"plots/plot_{i+1}")

if __name__=="__main__": 
    main() 

    

    