from .gaussian_measure import GaussianMeasure

def empirical_data_error_measure(model_measure, forward_operator, n_samples=10, scale_factor=1.0):
    """
    Generate an empirical data error measure based on samples from a measure on the model space. Useful for when you need
    to define a reasonable data error measure for synthetic testing, and need the covariance matrix to be easily accessible.
    
    Args:
        model_measure: The measure on the model space used as a basis for the error measure (e.g., the model prior measure)
        forward_operator: Linear operator mapping from model space to data space (e.g., operator B)
        n_samples: Number of samples to generate for computing statistics (default: 10)
        scale_factor: Scaling factor for the standard deviations (default: 1.0)

    Returns:
        inf.GaussianMeasure: Data error measure with empirically determined covariance
    """
    # Generate samples in data space by pushing forward model samples
    data_samples = model_measure.affine_mapping(operator=forward_operator).samples(n_samples)
    data_space = forward_operator.codomain
    
    # Remove the mean from each sample
    total = data_space.zero
    for sample in data_samples:
        total = data_space.add(total, sample)
    mean = data_space.multiply(1.0 / n_samples, total)
    zeroed_samples = [data_space.multiply(scale_factor, data_space.subtract(data_sample, mean)) for data_sample in data_samples]

    # Create and return the Gaussian measure from the zeroed samples
    return GaussianMeasure.from_samples(forward_operator.codomain, zeroed_samples)