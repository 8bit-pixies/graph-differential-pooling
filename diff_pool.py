import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-15


class DiffPool(nn.Module):
    """
    A fully connected diff pool with soft assignment (up and down)
    """

    def __init__(self, input_size, output_size, init_nodes, pool_layers=[2, 1]):
        super().__init__()
        # discrim_size was 32 or 64
        self.input_size = input_size
        self.output_size = output_size
        self.init_nodes = init_nodes
        self.pool_layers = pool_layers

        pool_layers = [init_nodes] + pool_layers
        self.layers = []
        for in_nodes, out_nodes in zip(pool_layers[:-1], pool_layers[1:]):
            self.layers.append(torch.nn.Parameter(torch.rand(1, in_nodes, out_nodes)))

    def forward(self, x):
        # go one direction
        for s in self.layers:
            print("x in", x.shape)
            print("s", s.shape)
            x = torch.matmul(torch.softmax(s, dim=-1).transpose(1, 2), x)
            print("x out", x.shape)
        return x

    def reverse(self, x):
        layer_reverse = []
        for idx, s in enumerate(self.layers):
            # print("x in", x.shape)
            # print("s", s.shape)
            x = torch.matmul(torch.softmax(s, dim=-1).transpose(1, 2), x)
            # print("x out", x.shape)

            # go back up
            x_prev = x.detach().clone()
            print("idx", idx)
            for s_prev in self.layers[: idx + 1][::-1]:
                x_prev = torch.matmul(torch.softmax(s_prev, dim=1), x_prev)
                print(x_prev.shape)
            layer_reverse.append(x_prev)
        return layer_reverse


in_size, out_size, start_clust = 5, 4, 8
pool_layers = [4, 2, 1]

x_input = torch.randn((1, start_clust, in_size))
diffpool = DiffPool(in_size, out_size, start_clust, pool_layers)
x_prev = diffpool.reverse(x_input)


def dense_diff_pool(x, adj, s, mask=None):
    r"""Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper
    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}
        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})
    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns pooled node feature matrix, coarsened adjacency matrix and two
    auxiliary objectives: (1) The link prediction loss
    .. math::
        \mathcal{L}_{LP} = {\| \mathbf{A} -
        \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
        \|}_F,
    and the entropy regularization
    .. math::
        \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).
    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
            \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss


def test_dense_diff_pool():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = torch.randn((batch_size, num_nodes, channels))
    adj = torch.rand((batch_size, num_nodes, num_nodes))
    s = torch.randn((1, num_nodes, num_clusters))
    mask = torch.randint(0, 2, (batch_size, num_nodes), dtype=torch.bool)

    x_prime, adj_prime, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)

    # rearrange..
    # cluster classification weights
    out_size, node_features = 3, 16
    w = torch.randn((out_size, node_features))
    clust_prediction = F.linear(x_prime, w)
    s_proba = torch.softmax(s, dim=-1)

    # remap cluster predictions back up to the node level, using s (inverse?)
    x_next = torch.matmul(torch.softmax(s, dim=2).transpose(1, 2), x)

    # aim to build a matrix s such that the mapping in both directions
    # minimises reconstruction error...

    # this allows for unpooling of the other direction; that is order sensitive
    # this is based on the unpooling paper:
    # "Learning Two-View Correspondences and Geometry using Order-Aware Network"
    x_prev = torch.matmul(torch.softmax(s, dim=1), x_next)

    # now remapping dense proba back to the original node order...
    # you can double up or share weights - depends on intent
    x_prev_prediction = torch.matmul(torch.softmax(s, dim=1), clust_prediction)
