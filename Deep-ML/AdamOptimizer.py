import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
	"""
	Update parameters using the Adam optimizer.
	Adjusts the learning rate based on the moving averages of the gradient and squared gradient.
	:param parameter: Current parameter value
	:param grad: Current gradient
	:param m: First moment estimate
	:param v: Second moment estimate
	:param t: Current timestep
	:param learning_rate: Learning rate (default=0.001)
	:param beta1: First moment decay rate (default=0.9)
	:param beta2: Second moment decay rate (default=0.999)
	:param epsilon: Small constant for numerical stability (default=1e-8)
	:return: tuple: (updated_parameter, updated_m, updated_v)
	"""
    # update biased first and second moment estimates
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # compute bias-corrected estimates
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # update parameters
    parameter = parameter - learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon))
	return np.round(parameter,5), np.round(m,5), np.round(v,5)

	# problem link : https://www.deep-ml.com/problems/87