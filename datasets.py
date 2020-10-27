import torch
from torch.utils.data import Dataset

import numpy
import pandas
from nltk import word_tokenize

from utils import get_max_length, create_glove_dict


class dataset(Dataset):
    def __init__(self, path_to_glove='glove.6B.200d.txt',
                 embedding_dim=200, prep_Data_from = 'train', purpose='train_model'):
        """
        NOTE-
        Beware of NaNs, drop them beforehand
        Dataset is the Liar Dataset. The description of the data can be found here -
        "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection - https://arxiv.org/abs/1705.00648"
        Download the dataset from - https://github.com/Tariq60/LIAR-PLUS
        Find the Glove vectors at https://nlp.stanford.edu/projects/glove/ and download the 822MB one.
        It contains 50d,100d, 200d and 300d vectors.
        300d with 400K vocab takes around 1.5GB RAM, choose file according to your system.
        We have prepared test cases using the 200d vectors. 
        :param path_to_glove: path to the desired glove vector file. File would be a .txt file
        :param embedding_dim: The dimension of vector you are choosing.
        :param prep_Data_from: Chose file from which you wanna prep data. 
        :param purpose: This is only used by the test.py file. This parameter should not concern you. When making your dataloaders, DO NOT pass this parameter. 
        """
        assert prep_Data_from in ['train', 'test', 'val']
        assert purpose in ['train_model', 'test_class']
        
        if purpose == 'train_model':
            path_to_train = 'train2.tsv'
            path_to_val = 'val2.tsv'
            path_to_test = 'test2.tsv'
        else:
            path_to_train = 'sample_train.tsv'
            path_to_test = 'sample_test.tsv'
            path_to_val = 'sample_val.tsv'

        train_Dataframe = pandas.read_csv(path_to_train, sep='\t', header=None).dropna()
        test_Dataframe = pandas.read_csv(path_to_test, sep='\t', header=None).dropna()
        val_Dataframe = pandas.read_csv(path_to_val, sep='\t', header=None).dropna()

        self.embeddings = create_glove_dict(path_to_glove)
        self.embedding_dim = embedding_dim
        self.dataframe = pandas.concat([train_Dataframe, test_Dataframe, val_Dataframe])

        self.justification_max = get_max_length(self.dataframe, 15)
        self.statement_max = get_max_length(self.dataframe, 3)

        if prep_Data_from == 'train':
            self.dataframe = train_Dataframe
        elif prep_Data_from == 'val':
            self.dataframe = val_Dataframe
        elif prep_Data_from == 'test':
            self.dataframe = test_Dataframe

        del train_Dataframe, test_Dataframe, val_Dataframe

        self.labels = {"true": 0,
                       "mostly-true": 1,
                       "half-true": 2,
                       "barely-true": 3,
                       "false": 4,
                       "pants-fire": 5}

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx): # 1.25 Mark
        data_dict = {}

        #step 1
        #load statement
        statement_data = self.dataframe.iloc[idx, 3]
        statement_data = statement_data.lower()
        statement = []
        statement.extend(word_tokenize(statement_data))

        # print(statement_data)

        #load justification
        justification_data = self.dataframe.iloc[idx, 15]
        justification_data = justification_data.lower()
        justification = []
        justification.extend(word_tokenize(justification_data))

        # step 2 -- convert label to torch tensor
        label_str = self.dataframe.iloc[idx, 2]
        # label = numpy.zeros(6)
        # numpy.put(label, self.labels[label_str], 1)
        label = self.labels[label_str]
        label_tensor = torch.tensor(label)

        # create the numpy word vectors
        # embedding_dimension * max_length
        vectorized_statement = numpy.zeros((self.embedding_dim, self.statement_max),dtype = "float32")
        vectorized_justification = numpy.zeros((self.embedding_dim, self.justification_max), dtype = "float32")

        #step 4 populate the numpy arrays
        for pos, word in enumerate(statement):
            try:
                # print(word)
                # print(self.embeddings[word])
                vectorized_statement[:, pos] = self.embeddings[word].copy() # store the embeddings in the pos th column
            except KeyError:
                print('Word not in Vocab. Placing it at origin')
                continue
        
        for pos, word in enumerate(justification):
            try:
                vectorized_justification[:, pos] = self.embeddings[word].copy() # copying the embeddings int the posth column
            except KeyError:
                print('Word not in Vocab. Placing it at origin')
                continue

        # 9 - Barely true counts
        # 10 - False counts
        # 11 half true counts
        # 12 mostly true counts
        # 13 pants on fire count
        #step 5, 6 -- convert to torch tensors
        statement_tensor = torch.tensor(vectorized_statement, dtype = torch.float32)
        justification_tensor = torch.tensor(vectorized_justification, dtype = torch.float32)
        
        #step 7- credit history column 9 - 13
        credit_history = numpy.zeros(5, dtype = 'float32')
        credit_history[0] = self.dataframe.iloc[idx, 9]
        credit_history[1] = self.dataframe.iloc[idx, 10]
        credit_history[2] = self.dataframe.iloc[idx, 11]
        credit_history[3] = self.dataframe.iloc[idx, 12]
        credit_history[4] = self.dataframe.iloc[idx, 13]

        credit_history_tensor = torch.from_numpy(credit_history)

        # print(statement_tensor, justification_tensor, label_tensor, credit_history_tensor)

        #step 8 create the dict
        data_dict["statement"] = statement_tensor
        data_dict["justification"] = justification_tensor
        data_dict["label"] = label_tensor
        data_dict["credit_history"] = credit_history_tensor

        del vectorized_statement, vectorized_justification, label, credit_history  # just free up some memory
        return data_dict

    def get_max_lenghts(self):
        return self.statement_max, self.justification_max

    def get_Data_shape(self):
        return self.dataframe.shape