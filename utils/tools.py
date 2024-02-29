import torch


def compute_output_dim(model, input_shape=(3, 51, 51)):
    """Compute the number of output neurons of convolutional layers
    """
    # Create a dummy input with the given shape
    dummy_input = torch.randn(1, *input_shape)
    # Forward pass the dummy input through the model
    dummy_output = model(dummy_input)
    # Return the output shape
    return dummy_output.shape[-1]