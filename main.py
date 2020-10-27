# 0.75 Marks. 
# To test your trainer and  arePantsonFire class, Just create random tensor and see if everything is working or not.  
from torch.utils.data import DataLoader

# Your code goes here.
import datasets
from datasets import dataset
from Encoder import Encoder
from Attention import PositionFeedforward, MultiHeadAttention
from LiarLiar import arePantsonFire
from trainer import trainer

liar_dataset_train = dataset(path_to_glove='glove.6B.200d.txt', embedding_dim=200, prep_Data_from = 'train', purpose='train_model')
liar_dataset_val = dataset(path_to_glove='glove.6B.200d.txt', embedding_dim=200, prep_Data_from = 'val', purpose='train_model')
dataloader_train = DataLoader(dataset=liar_dataset_train, batch_size=50)
dataloader_val = DataLoader(dataset = liar_dataset_val, batch_size= 25)
statemnet_max, justification_max = liar_dataset_train.get_max_lenghts()

statement_encoder = Encoder(5, 512)
justification_encoder = Encoder(5, 512)

multiheadAttention = MultiHeadAttention(512, 32)
positionFeedForward = PositionFeedforward(512, 2048)

model = arePantsonFire(statement_encoder, justification_encoder, multiheadAttention, positionFeedForward, 512, statemnet_max, justification_max, 200)

trainer(model, dataloader_train, dataloader_val, 1, '/home/rajat/Desktop/NNFL Assign/Assignment Files','/home/rajat/Desktop/NNFL Assign/Assignment Files', 100, 1, 1, 'cpu')

# Do not change module_list , otherwise no marks will be awarded
module_list = [liar_dataset_train, liar_dataset_val, dataloader_train, dataloader_val, statement_encoder, justification_encoder, multiheadAttention, positionFeedForward, model]
del  liar_dataset_val, liar_dataset_train, dataloader_train, dataloader_val


liar_dataset_test = dataset(prep_Data_from='test')
test_dataloader = DataLoader(dataset=liar_dataset_test, batch_size=1)
infer(model=model, dataloader=test_dataloader)
