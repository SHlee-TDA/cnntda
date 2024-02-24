import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x


class ModalFusionUnit(nn.Module):
    """
    A module to fuse feature vectors from multiple modalities using learnable weights.
    
    This unit applies a softmax to the learnable weights to ensure they sum up to 1, 
    representing the relative importance of each modality. The output is the weighted 
    sum of the input feature vectors from each modality.

    Attributes:
        weights (torch.nn.Parameter): Learnable weights for each modality.
        feature_length (int): The length of the feature vectors from each modality.

    Args:
        num_modals (int): The number of modalities to be fused.
        feature_length (int): The length of the feature vectors from each modality.
    """

    def __init__(self, num_modals, feature_length):
        super(ModalFusionUnit, self).__init__()
        # Initialize the weights to equal values; softmax will make them sum up to 1
        self.weights = nn.Parameter(torch.zeros(num_modals))

        self.feature_length = feature_length

    def forward(self, *features):
        """
        Forward pass of the ModalFusionUnit.

        The method takes variable number of feature vectors (one for each modality),
        applies the learned weights, and returns the weighted sum.

        Args:
            *features: A variable number of tensors representing feature vectors 
                       from different modalities. Each tensor is of shape 
                       [batch_size, feature_length].

        Returns:
            torch.Tensor: The resulting fused feature vector of shape 
                          [batch_size, feature_length].
        """
        # Check that all feature vectors have the same length
        assert all(feature.shape[1] == self.feature_length for feature in features), \
            "All feature vectors must have the same length"

        # Stack the feature vectors along a new dimension
        stacked_features = torch.stack(features, dim=1)  # Shape: [batch_size, num_modals, feature_length]

        # Apply softmax to the weights and calculate the weighted sum of features
        normalized_weights = F.softmax(self.weights, dim=0)
        weighted_sum = torch.einsum('bmn,m->bn', stacked_features, normalized_weights)

        return weighted_sum


class CNNTDAPlus(nn.Module):
    def __init__(self, 
                 image_shape,
                 top_shape,
                 image_model, 
                 topology_model, 
                 classifier, 
                 projection_dim=256):
        super(CNNTDAPlus, self).__init__()
        self.projection_dim = projection_dim
        
        # Image subnetwork
        self.image_model = image_model
        self.input_dim_for_img_repr = compute_output_dim(self.image_model, image_shape)
        self.proj_image = ProjectionLayer(input_dim=self.input_dim_for_img_repr, 
                                          output_dim=self.projection_dim)
        # Topological feature subnetwork
        self.topology_model = topology_model
        self.input_dim_for_top_repr = compute_output_dim(self.topology_model, top_shape)
        self.proj_top = ProjectionLayer(input_dim=self.input_dim_for_top_repr, output_dim=projection_dim)

        # Classifier
        self.fusion = ModalFusionUnit(num_modals=2, feature_length=projection_dim)
        self.classifier = classifier

    def forward(self, image_data, topology_data):
        # Get representations from each subnetwork
        image_repr = self.proj_image(self.image_model(image_data))
        topology_repr = self.proj_top(self.topology_model(topology_data))

        # Concatenate the representations
        combined_repr = self.fusion(image_repr, topology_repr)
        
        # Pass through the classifier
        output = self.classifier(combined_repr)

        return output
    
    def get_modal_importance(self):
        return F.softmax(self.fusion.weights, dim=0)

    @property
    def module(self):
        """
        If the model is being used with DataParallel, 
        this property will return the actual model.
        """
        return self._modules.get('module', self)
    

def compute_output_dim(model, input_shape=(3, 51, 51)):
    """Compute the number of output neurons of convolutional layers
    """
    # Create a dummy input with the given shape
    dummy_input = torch.randn(1, *input_shape)
    # Forward pass the dummy input through the model
    dummy_output = model(dummy_input)
    # Return the output shape
    return dummy_output.shape[-1]