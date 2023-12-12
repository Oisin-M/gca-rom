import torch
from torch import nn
from gca_rom import gca, gbm, scaling


class HyperParams:
    """Class that holds the hyperparameters for the autoencoder model.

    Args:
        sparse_method (str): The method to use for sparsity constraint.
        rate (int): Amount of data used in training.
        seed (int): Seed for the random number generator.
        bottleneck_dim (int): The dimension of the bottleneck layer.
        tolerance (float): The tolerance value for stopping the training.
        lambda_map (float): The weight for the map loss.
        learning_rate (float): The learning rate for the optimizer.
        ffn (int): The number of feed-forward layers.
        in_channels (int): The number of input channels.
        hidden_channels (list): The number of hidden channels for each layer.
        act (function): The activation function to use.
        nodes (int): The number of nodes in each hidden layer.
        skip (int): The number of skipped connections.
        layer_vec (list): The structure of the network.
        net_name (str): The name of the network.
        scaler_name (str): The name of the scaler used for preprocessing.
        weight_decay (float): The weight decay for the optimizer.
        max_epochs (int): The maximum number of epochs to run training for.
        miles (list): The miles for learning rate update in scheduler.
        gamma (float): The gamma value for the optimizer.
        num_nodes (int): The number of nodes in the network.
        scaling_type (int): The type of scaling to use for preprocessing.
        net_dir (str): The directory to save the network in.
        cross_validation (bool): Whether to perform cross-validation.
    """

    def __init__(self, argv):
        self.net_name = argv[0]
        self.variable = argv[1]
        self.scaling_type = int(argv[2])
        self.scaler_number = int(argv[3])
        _, self.scaler_name = scaling.scaler_functions(self.scaler_number)
        self.skip = int(argv[4])
        self.rate = int(argv[5])
        self.sparse_method = 'L1_mean'
        self.ffn = int(argv[6])
        self.nodes = int(argv[7])
        self.bottleneck_dim = int(argv[8])
        self.lambda_map = float(argv[9])
        self.in_channels = int(argv[10])
        self.seed = 10
        self.tolerance = 1e-6
        self.learning_rate = 0.001
        self.hidden_channels = [1]*self.in_channels
        self.act = torch.tanh
        self.layer_vec=[argv[11], self.nodes, self.nodes, self.nodes, self.nodes, self.bottleneck_dim]
        self.net_run = '_' + self.scaler_name
        self.weight_decay = 0.00001
        self.max_epochs = argv[12]
        self.miles = []
        self.gamma = 0.0001
        self.num_nodes = 0
        self.GBM = argv[13]
        self.net_dir = './' + self.net_name + '/' + self.net_run + '/' + self.variable + '_' + self.net_name + '_lmap' + str(self.lambda_map) + '_btt' + str(self.bottleneck_dim) \
                            + '_seed' + str(self.seed) + '_lv' + str(len(self.layer_vec)-2) + '_hc' + str(len(self.hidden_channels)) + '_nd' + str(self.nodes) \
                            + '_ffn' + str(self.ffn) + '_skip' + str(self.skip) + '_lr' + str(self.learning_rate) + '_sc' + str(self.scaling_type) + '_rate' + str(self.rate) + '_gbm' + str(self.GBM) + '/'
        self.cross_validation = True



class Net(torch.nn.Module):
    """
    Class Net
    ---------

    A PyTorch neural network class which consists of encoder, decoder and mapping modules.

    Attributes
    ----------
    encoder : gca.Encoder
        An encoder module from the gca module.
    decoder : gca.Decoder
        A decoder module from the gca module.
    act_map : HyperParams.act
        The activation map specified in the HyperParams.
    layer_vec : HyperParams.layer_vec
        The layer vector specified in the HyperParams. Shape of the layers for the parameter space mapping. 
    steps : int
        Number of layers for the parameter mapping MLP.
    maptovec : nn.ModuleList
        A list of linear layers for mapping.

    Methods
    -------
    solo_encoder(data)
        Encodes the input data using the encoder module.
        Returns the encoded representation.
    solo_decoder(x, data)
        Decodes the encoded representation and the input data using the decoder module.
        Returns the decoded output.
    mapping(x)
        Maps the input using the linear modules in maptovec.
        Returns the mapped output.
    forward(data, parameters)
        Runs a forward pass through the network using the input data and parameters.
        Returns the decoded output, encoded representation, and estimated encoded representation.
    """

    def __init__(self, HyperParams, domain=None):
        super().__init__()
        if HyperParams.GBM:
            if domain is None:
                raise Exception("Must specify domain for GBM.")
            else:
                interpolation_grid = self.generate_interpolation_grid(domain, HyperParams.bottleneck_dim)
            self.encoder = gbm.Encoder(HyperParams.hidden_channels, HyperParams.bottleneck_dim, HyperParams.num_nodes, ffn=HyperParams.ffn, skip=HyperParams.skip, interpolation_grid=interpolation_grid)
            self.decoder = gbm.Decoder(HyperParams.hidden_channels, HyperParams.bottleneck_dim, HyperParams.num_nodes, ffn=HyperParams.ffn, skip=HyperParams.skip, interpolation_grid=interpolation_grid)
        else:
            self.encoder = gca.Encoder(HyperParams.hidden_channels, HyperParams.bottleneck_dim, HyperParams.num_nodes, ffn=HyperParams.ffn, skip=HyperParams.skip)
            self.decoder = gca.Decoder(HyperParams.hidden_channels, HyperParams.bottleneck_dim, HyperParams.num_nodes, ffn=HyperParams.ffn, skip=HyperParams.skip)

        self.act_map = HyperParams.act
        self.layer_vec = HyperParams.layer_vec
        self.steps = len(self.layer_vec) - 1

        self.maptovec = nn.ModuleList()
        for k in range(self.steps):
            self.maptovec.append(nn.Linear(self.layer_vec[k], self.layer_vec[k+1]))

    def generate_interpolation_grid(self, domain, bottleneck_dim):
        import shapely.geometry as geo
        import numpy as np
        if domain=='stokes':
            data = [
                    [(1,2), (1,1), (2,1), (1,2)],
                    [(2,1), (2,2), (1,2), (2,1)],
                    [(0,3), (0,2), (1,2), (0,3)],
                    [(1,2), (1,3), (0,3), (1,2)],
                    [(0,2), (0,1), (1,1), (0,2)],
                    [(1,1), (1,2), (0,2), (1,1)],
                    [(0,1), (0,0), (1,0), (0,1)],
                    [(1,0), (1,1), (0,1), (1,0)]
            ]
            domain_list = [geo.Polygon(np.array(x).astype('float')) for x in data]

            def random_point_in_shp(shp):
                within = False
                while not within:
                    x = np.random.uniform(shp.bounds[0], shp.bounds[2])
                    y = np.random.uniform(shp.bounds[1], shp.bounds[3])
                    within = shp.contains(geo.Point(x, y))
                return (x,y)

            # Randomly sample from each subdomain
            reference_pts = []
            for domain in domain_list:
                for i in range(bottleneck_dim // len(domain_list)):
                    reference_pts.append(random_point_in_shp(domain))
            for i in range(bottleneck_dim % len(domain_list)):
                reference_pts.append(random_point_in_shp(domain_list[i]))

            interpolation_grid = torch.tensor(reference_pts)
        else:
            raise Exception("GBM currently only implemented for Stokes problem.")
        return interpolation_grid

    def solo_encoder(self, data):
        x = self.encoder(data)
        return x

    def solo_decoder(self, x, data):
        x = self.decoder(x, data)
        return x

    def mapping(self, x):
        idx = 0
        for layer in self.maptovec:
            if (idx==self.steps): x = layer(x)
            else: x = self.act_map(layer(x))
            idx += 1
        return x

    def forward(self, data, parameters):
        z = self.solo_encoder(data)
        z_estimation = self.mapping(parameters)
        x = self.solo_decoder(z, data)
        # x = self.solo_decoder(z_estimation, data)
        return x, z, z_estimation
