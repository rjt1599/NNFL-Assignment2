import torch
import torch.nn as nn

from ConvS2S import ConvEncoder
from Attention import MultiHeadAttention, PositionFeedforward

class Encoder(nn.Module): # 1 Mark
    def __init__(self, conv_layers, hidden_dim, feed_forward_dim=2048):
        super(Encoder, self).__init__()
        # Your code here
        # Encoder class, specify the number of attention heads as 16. 

        self.conv = ConvEncoder(hidden_dim, conv_layers)
        self.attention = MultiHeadAttention(hidden_dim, 16)
        self.feed_forward = PositionFeedforward(hidden_dim, feed_forward_dim)


    def forward(self, input):
        """
        Forward Pass of the Encoder Class
        :param input: Input Tensor for the forward pass. 
        """
        # Your code here
        out1 = self.conv.forward(input)
        out2 = self.attention.forward(out1, out1, out1)
        out3 = self.feed_forward.forward(out2)
        return out3