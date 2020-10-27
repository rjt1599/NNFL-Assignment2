import torch
import torch.nn as nn

from Attention import MultiHeadAttention, PositionFeedforward
from Encoder import Encoder

class arePantsonFire(nn.Module):

    # def __init__(self, sentence_encoder: Encoder, explanation_encoder: Encoder, multihead_Attention: MultiHeadAttention,
    #              position_Feedforward: PositionFeedforward, hidden_dim: int, max_length_sentence,
    #              max_length_justification, input_dim, device='cuda:0'):

    def __init__(self, sentence_encoder, explanation_encoder, multihead_Attention,
                 position_Feedforward, hidden_dim, max_length_sentence,
                 max_length_justification, input_dim, device='cpu'):
        """
        If you wish to shift on cpu pass device as 'cpu'
        """

        super(arePantsonFire, self).__init__()
        self.device = device

        self.sentence_pos_embedding = nn.Embedding(max_length_sentence, hidden_dim)
        self.justification_pos_embedding = nn.Embedding(max_length_justification, hidden_dim)

        self.sentence_encoder = sentence_encoder
        self.explanation_encoder = explanation_encoder
        self.attention = multihead_Attention
        self.position_feedforward = position_Feedforward

        self.upscale_conv, self.first_conv, self.flatten_conv = self.get_convolutions(input_dim=input_dim, hidden_dim=hidden_dim)
        self.linear1, self.linear2, self.bilinear, self.classifier = self.get_linears_layers(max_length_sentence=max_length_sentence)

    def forward(self, sentence, justification, credit_history): # 1 Marks

        sentence_pos = torch.arange(0, sentence.shape[2]).unsqueeze(0).repeat(sentence.shape[0],1).to(self.device).long()
        justification_pos = torch.arange(0, justification.shape[2]).unsqueeze(0).repeat(justification.shape[0], 1).to(self.device).long()

        sentence = self.upscale_conv(sentence)
        sentence = sentence + self.sentence_pos_embedding(sentence_pos).permute(0, 2, 1)

        justification = self.upscale_conv(justification)
        justification = justification + self.justification_pos_embedding(justification_pos).permute(0, 2, 1)
        
        # Your code goes here
        #step1 encode the sentence and justification
        sentence = self.sentence_encoder.forward(sentence)
        justification = self.explanation_encoder.forward(justification)

        #step2
        attention_output = self.attention.forward(sentence, justification, justification)

        #step3
        out = self.position_feedforward.forward(attention_output)

        #step4
        out = nn.functional.relu(self.first_conv(out))

        #step5
        out = self.flatten_conv(out)

        #step6
        out = out.view(out.size(0), -1)

        #step 7
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.bilinear(out, credit_history)

        #step 8
        out = self.classifier(out)

        return out

    def get_convolutions(self, input_dim, hidden_dim): # 0.5 Marks
        # Your code here
        upscale_conv = nn.Conv1d(in_channels=input_dim, out_channels= hidden_dim, kernel_size=1, stride=1, padding = 0)
        first_conv = nn.Conv1d(in_channels=hidden_dim, out_channels= hidden_dim // 2, kernel_size=3, stride=1, padding = 1)
        flatten_conv = nn.Conv1d(in_channels=hidden_dim // 2, out_channels= 1, kernel_size=5, stride=1, padding = 2)
        return upscale_conv, first_conv, flatten_conv

    def get_linears_layers(self, max_length_sentence): # 0.5 Marks
        # Your code goes here
        linear1 = nn.Linear(in_features=max_length_sentence, out_features=max_length_sentence//4)
        linear2 = nn.Linear(in_features=max_length_sentence//4, out_features=6)
        bilinear = nn.Bilinear(in1_features = 6, in2_features = 5,out_features = 12, bias = True)
        classifier = nn.Linear(in_features = 12, out_features = 6)
        return linear1, linear2, bilinear, classifier