import torch
import torch.nn as nn
import torch.nn.functional as F


class Distance(nn.Module):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def forward(self, inputs):
        return torch.sqrt(torch.sum(inputs ** 2, dim=-1) + 1e-9)


class Mask(nn.Module):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    """
    def forward(self, inputs, mask=None):
        if mask is None: # If no true label provided, mask by max length of capsules
            lengths = torch.sqrt(torch.sum(inputs ** 2, dim=-1))
            mask = F.one_hot(torch.argmax(lengths, dim=1), num_classes=inputs.size(1)).float()
        else:
            mask = mask.float()

        masked = inputs * mask.unsqueeze(-1)
        return masked.view(masked.size(0), -1)

def squash(vectors, dim=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param dim: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = torch.sum(vectors ** 2, dim=dim, keepdim=True)
    scale = s_squared_norm / (1 + s_squared_norm) / torch.sqrt(s_squared_norm + 1e-9)
    return scale * vectors


class CapsuleLayer(nn.Module):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

        self.W = nn.Parameter(torch.randn(1, num_capsule, dim_capsule, dim_capsule))

    def forward(self, inputs):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        batch_size = inputs.size(0)
        input_num_capsule = inputs.size(1)
        input_dim_capsule = inputs.size(2)

        # Expand dimensions to prepare for multiplication
        inputs_expand = inputs.unsqueeze(1).expand(batch_size, self.num_capsule, input_num_capsule, input_dim_capsule)

        # Repeat W for batch_size
        W = self.W.expand(batch_size, self.num_capsule, input_dim_capsule, self.dim_capsule)

        # Matrix multiplication
        inputs_hat = torch.einsum('bijp,bjkl->bikl', inputs_expand, W)

        # Routing algorithm
        b = torch.zeros(batch_size, self.num_capsule, input_num_capsule).to(inputs.device)

        for i in range(self.routings):
            c = F.softmax(b, dim=1)
            outputs = squash(torch.einsum('bijk,bij->bik', inputs_hat, c))

            if i < self.routings - 1:
                b = b + torch.einsum('bikj,bik->bikj', inputs_hat, outputs)

        return outputs


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    def __init__(self, dim_capsule, n_channels, kernel_size, stride, padding):
        super(PrimaryCapsule, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=dim_capsule*n_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.dim_capsule = dim_capsule

    def forward(self, inputs):
        output = self.conv(inputs)
        output = output.view(output.size(0), -1, self.dim_capsule)
        return squash(output)
