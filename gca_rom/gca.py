import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class Encoder(torch.nn.Module):
    """
    Encoder Class

    The Encoder class is a subclass of torch.nn.Module that implements a deep neural network for encoding graph data.

    It uses the Gaussian Mixture convolution (GMMConv) module to extract features from the graph structure and node features.
    The encoding is then passed through a feed-forward neural network with two fully connected layers to produce the final encoding.

    Arguments:
    hidden_channels (list): A list of hidden channel sizes for each layer of the GMMConv module.
    bottleneck (int): Size of the bottleneck layer in the feed-forward neural network.
    input_size (int): Size of the node features.
    ffn (int): Size of the intermediate layer in the feed-forward neural network.
    skip (bool): If True, the input node features will be concatenated with the GMMConv output at each layer.
    act (function): Activation function used in the GMMConv layers and feed-forward neural network. Defaults to F.elu.

    Methods:
    encoder(data): Encodes the graph data using the GMMConv module and feed-forward neural network.
    reset_parameters(): Resets the parameters of the GMMConv layers and feed-forward neural network.
    forward(data): A convenience function that calls the encoder method.
    """

    def __init__(self, hidden_channels, bottleneck, input_size, ffn, skip, act=F.elu, conv='GMMConv'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = len(self.hidden_channels)
        self.act = act
        self.ffn = ffn
        self.skip = skip
        self.bottleneck = bottleneck
        self.input_size = input_size
        self.conv = conv

        self.down_convs = torch.nn.ModuleList()
        for i in range(self.depth-1):
            if self.conv=='GMMConv':
                self.down_convs.append(gnn.GMMConv(self.hidden_channels[i], self.hidden_channels[i+1], dim=1, kernel_size=5))
            elif self.conv=='ChebConv':
                self.down_convs.append(gnn.ChebConv(self.hidden_channels[i], self.hidden_channels[i+1], K=5))
            elif self.conv=='GCNConv':
                self.down_convs.append(gnn.GCNConv(self.hidden_channels[i], self.hidden_channels[i+1]))
            elif self.conv=='GATConv':
                self.down_convs.append(gnn.GATConv(self.hidden_channels[i], self.hidden_channels[i+1]))
            else:
                raise NotImplementedError('Invalid convolution selected. Please select one of [GMMConv, ChebConv, GCNConv, GATConv]')

        self.fc_in1 = nn.Linear(self.input_size*self.hidden_channels[-1], self.ffn)
        self.fc_in2 = nn.Linear(self.ffn, self.bottleneck)
        self.reset_parameters()

    def encoder(self, data):
        x = data.x
        idx = 0
        for layer in self.down_convs:
            if self.conv in ['GMMConv', 'ChebConv', 'GCNConv']:
                x = self.act(layer(x, data.edge_index, data.edge_weight))
            elif self.conv in ['GATConv']:
                x = self.act(layer(x, data.edge_index, data.edge_attr))
            if self.skip:
                x = x + data.x
            idx += 1

        x = x.reshape(data.num_graphs, self.input_size * self.hidden_channels[-1])
        x = self.act(self.fc_in1(x))
        x = self.fc_in2(x)
        return x

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.kaiming_uniform_(param)

    def forward(self,data):
        x = self.encoder(data)
        return x


class Decoder(torch.nn.Module):
    """
    Class Decoder
    A torch.nn.Module subclass for the decoder part of a neural network.

    Attributes:
        hidden_channels (list of ints): A list of hidden channel sizes.
        depth (int): The length of hidden_channels list.
        act (function): Activation function to use.
        ffn (int): Size of output after the first linear layer.
        skip (bool): Whether to add skip connections.
        bottleneck (int): Size of bottleneck layer.
        input_size (int): Size of input data.
        fc_out1 (torch.nn.Linear): Linear layer from bottleneck to ffn.
        fc_out2 (torch.nn.Linear): Linear layer from ffn to input_size * hidden_channels[-1].
        up_convs (torch.nn.ModuleList): A list of GMMConv layers.

    Methods:
        decoder(self, x, data):
            Decodes the input data x and returns the output.

        reset_parameters(self):
            Resets the parameters of the up_convs layer.

        forward(self, x, data):
            Performs a forward pass on the input data x and returns the output.
    """

    def __init__(self, hidden_channels, bottleneck, input_size, ffn, skip, act=F.elu, conv='GMMConv'):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.depth = len(self.hidden_channels)
        self.act = act
        self.ffn = ffn
        self.skip = skip
        self.bottleneck = bottleneck
        self.input_size = input_size
        self.conv = conv

        self.fc_out1 = nn.Linear(self.bottleneck, self.ffn)
        self.fc_out2 = nn.Linear(self.ffn, self.input_size * self.hidden_channels[-1])

        self.up_convs = torch.nn.ModuleList()
        for i in range(self.depth-1):
            if self.conv=='GMMConv':
                self.up_convs.append(gnn.GMMConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2], dim=1, kernel_size=5))
            elif self.conv=='ChebConv':
                self.up_convs.append(gnn.ChebConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2], K=5))
            elif self.conv=='GCNConv':
                self.up_convs.append(gnn.GCNConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2]))
            elif self.conv=='GATConv':
                self.up_convs.append(gnn.GATConv(self.hidden_channels[self.depth-i-1], self.hidden_channels[self.depth-i-2]))
            else:
                raise NotImplementedError('Invalid convolution selected. Please select one of [GMMConv, ChebConv, GCNConv, GATConv]')
            
        
        self.reset_parameters()


    def decoder(self, x, data):
        x = self.act(self.fc_out1(x))
        x = self.act(self.fc_out2(x))
        h = x.reshape(data.num_graphs*self.input_size, self.hidden_channels[-1])
        x = h
        idx = 0
        for layer in self.up_convs:
            if self.conv in ['GMMConv', 'ChebConv', 'GCNConv']:
                x = layer(x, data.edge_index, data.edge_weight)
            elif self.conv in ['GATConv']:
                x = layer(x, data.edge_index, data.edge_attr)
            if (idx != self.depth - 2):
                x = self.act(x)
            if self.skip:
                x = x + h
            idx += 1
        return x

    def reset_parameters(self):

        for conv in self.up_convs:
            conv.reset_parameters()
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.kaiming_uniform_(param)


    def forward(self, x, data):
        x = self.decoder(x, data)
        return x
