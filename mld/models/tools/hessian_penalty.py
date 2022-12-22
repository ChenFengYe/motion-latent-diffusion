"""
## Adapted to work with our "batches"
Official PyTorch implementation of the Hessian Penalty regularization term from https://arxiv.org/pdf/2008.10599.pdf
Author: Bill Peebles
TensorFlow Implementation (GPU + Multi-Layer): hessian_penalty_tf.py
Simple Pure NumPy Implementation: hessian_penalty_np.py

Simple use case where you want to apply the Hessian Penalty to the output of net w.r.t. net_input:
>>> from hessian_penalty_pytorch import hessian_penalty
>>> net = MyNeuralNet()
>>> net_input = sample_input()
>>> loss = hessian_penalty(net, z=net_input)  # Compute hessian penalty of net's output w.r.t. net_input
>>> loss.backward()  # Compute gradients w.r.t. net's parameters

If your network takes multiple inputs, simply supply them to hessian_penalty as you do in the net's forward pass. In the
following example, we assume BigGAN.forward takes a second input argument "y". Note that we always take the Hessian
Penalty w.r.t. the z argument supplied to hessian_penalty:
>>> from hessian_penalty_pytorch import hessian_penalty
>>> net = BigGAN()
>>> z_input = sample_z_vector()
>>> class_label = sample_class_label()
>>> loss = hessian_penalty(net, z=net_input, y=class_label)
>>> loss.backward()
"""

import torch


def hessian_penalty(G, batch, k=2, epsilon=0.1, reduction=torch.max, return_separately=False, G_z=None, **G_kwargs):
    """
    Official PyTorch Hessian Penalty implementation.

    Note: If you want to regularize multiple network activations simultaneously, you need to
    make sure the function G you pass to hessian_penalty returns a list of those activations when it's called with
    G(z, **G_kwargs). Otherwise, if G returns a tensor the Hessian Penalty will only be computed for the final
    output of G.

    :param G: Function that maps input z to either a tensor or a list of tensors (activations)
    :param z: Input to G that the Hessian Penalty will be computed with respect to
    :param k: Number of Hessian directions to sample (must be >= 2)
    :param epsilon: Amount to blur G before estimating Hessian (must be > 0)
    :param reduction: Many-to-one function to reduce each pixel/neuron's individual hessian penalty into a final loss
    :param return_separately: If False, hessian penalties for each activation output by G are automatically summed into
                              a final loss. If True, the hessian penalties for each layer will be returned in a list
                              instead. If G outputs a single tensor, setting this to True will produce a length-1
                              list.
    :param G_z: [Optional small speed-up] If you have already computed G(z, **G_kwargs) for the current training
                iteration, then you can provide it here to reduce the number of forward passes of this method by 1
    :param G_kwargs: Additional inputs to G besides the z vector. For example, in BigGAN you
                     would pass the class label into this function via y=<class_label_tensor>

    :return: A differentiable scalar (the hessian penalty), or a list of hessian penalties if return_separately is True
    """
    if G_z is None:
        G_z = G(batch, **G_kwargs)
    z = batch["x"]
    rademacher_size = torch.Size((k, *z.size()))  # (k, N, z.size())
    dzs = epsilon * rademacher(rademacher_size, device=z.device)
    second_orders = []
    for dz in dzs:  # Iterate over each (N, z.size()) tensor in xs
        central_second_order = multi_layer_second_directional_derivative(G, batch, dz, G_z, epsilon, **G_kwargs)
        second_orders.append(central_second_order)  # Appends a tensor with shape equal to G(z).size()
    loss = multi_stack_var_and_reduce(second_orders, reduction, return_separately)  # (k, G(z).size()) --> scalar
    return loss


def rademacher(shape, device='cpu'):
    """Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    x = torch.empty(shape, device=device)
    x.random_(0, 2)  # Creates random tensor of 0s and 1s
    x[x == 0] = -1  # Turn the 0s into -1s
    return x


def multi_layer_second_directional_derivative(G, batch, dz, G_z, epsilon, **G_kwargs):
    """Estimates the second directional derivative of G w.r.t. its input at z in the direction x"""
    batch_plus = {**batch, "x": batch["x"] + dz}
    batch_moins = {**batch, "x": batch["x"] - dz}
    G_to_x = G(batch_plus, **G_kwargs)
    G_from_x = G(batch_moins, **G_kwargs)

    G_to_x = listify(G_to_x)
    G_from_x = listify(G_from_x)
    G_z = listify(G_z)

    eps_sqr = epsilon ** 2
    sdd = [(G2x - 2 * G_z_base + Gfx) / eps_sqr for G2x, G_z_base, Gfx in zip(G_to_x, G_z, G_from_x)]
    return sdd


def stack_var_and_reduce(list_of_activations, reduction=torch.max):
    """Equation (5) from the paper."""
    second_orders = torch.stack(list_of_activations)  # (k, N, C, H, W)
    var_tensor = torch.var(second_orders, dim=0, unbiased=True)  # (N, C, H, W)
    penalty = reduction(var_tensor)  # (1,) (scalar)
    return penalty


def multi_stack_var_and_reduce(sdds, reduction=torch.max, return_separately=False):
    """Iterate over all activations to be regularized, then apply Equation (5) to each."""
    sum_of_penalties = 0 if not return_separately else []
    for activ_n in zip(*sdds):
        penalty = stack_var_and_reduce(activ_n, reduction)
        sum_of_penalties += penalty if not return_separately else [penalty]
    return sum_of_penalties


def listify(x):
    """If x is already a list, do nothing. Otherwise, wrap x in a list."""
    if isinstance(x, list):
        return x
    else:
        return [x]


def _test_hessian_penalty():
    """
    A simple multi-layer test to verify the implementation.
    Function: G(z) = [z_0 * z_1, z_0**2 * z_1]
    Ground Truth Hessian Penalty: [4, 16 * z_0**2]
    """
    batch_size = 10
    nz = 2
    z = torch.randn(batch_size, nz)
    def reduction(x): return x.abs().mean()
    def G(z): return [z[:, 0] * z[:, 1], (z[:, 0] ** 2) * z[:, 1]]
    ground_truth = [4, reduction(16 * z[:, 0] ** 2).item()]
    # In this simple example, we use k=100 to reduce variance, but when applied to neural networks
    # you will probably want to use a small k (e.g., k=2) due to memory considerations.
    predicted = hessian_penalty(G, z, G_z=None, k=100, reduction=reduction, return_separately=True)
    predicted = [p.item() for p in predicted]
    print('Ground Truth: %s' % ground_truth)
    print('Approximation: %s' % predicted)  # This should be close to ground_truth, but not exactly correct
    print('Difference: %s' % [str(100 * abs(p - gt) / gt) + '%' for p, gt in zip(predicted, ground_truth)])


if __name__ == '__main__':
    _test_hessian_penalty()
