from .gaussian_measure import GaussianMeasure
import numpy as np

def empirical_data_error_measure(model_measure, forward_operator, n_samples=10, scale_factor=1.0, diagonal_only=False):
    """
    Generate an empirical data error measure based on samples from a measure on the model space. Useful for when you need
    to define a reasonable data error measure for synthetic testing, and need the covariance matrix to be easily accessible.
    
    Args:
        model_measure: The measure on the model space used as a basis for the error measure (e.g., the model prior measure)
        forward_operator: Linear operator mapping from model space to data space (e.g., operator B)
        n_samples: Number of samples to generate for computing statistics (default: 10)
        scale_factor: Scaling factor for the standard deviations (default: 1.0)
        diagonal_only: If True, compute standard deviations only for the diagonal elements (default: False)

    Returns:
        inf.GaussianMeasure: Data error measure with empirically determined covariance
    """
    # Generate samples in data space by pushing forward model samples
    data_samples = model_measure.affine_mapping(operator=forward_operator).samples(n_samples)

    if not diagonal_only:
    
        # Remove the mean from each sample
        mean = model_measure.affine_mapping(operator=forward_operator).expectation
        zeroed_samples = [scale_factor * (data_sample - mean) for data_sample in data_samples]
        
        return GaussianMeasure.from_samples(forward_operator.codomain, zeroed_samples)
    
    else:

        # Convert to numpy array for easier manipulation
        data_array = np.array([sample.data for sample in data_samples])    

        # Compute standard deviation for each data dimension (each tide gauge)
        std_devs = np.std(data_array, axis=0) * scale_factor
        
        return GaussianMeasure.from_standard_deviations(forward_operator.codomain, std_devs)