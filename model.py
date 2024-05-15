import torch #import pytorch library
import torch.nn as nn #import nn module from torch.nn library
import math #import math
import matplotlib.pyplot as plt #THis is too plt statistics regarding the model
import pandas as pd # PAndas foor manage csv data
import numpy as np #handling ragading pandas output arrays
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExcelToTensor_Features():
    """
    This is the class to tokenize the EEG signal data into required format.
    .csv file should have following format
    | Trial | Class | EEG_Channel_1 | EEG_Channel_2 | ... | EEG_Channel_N |
    -----------------------------------------------------------------------
    """
    def __init__(self,path,n_samples:int)->None:
        self.na_values =[" "] #value contain interms of regarding misiing values
        self.path = path #This is the path to csv file
        self.dataframe = pd.read_csv(path,na_values=self.na_values) #This create .csv dataframe
        self.df_class =self.dataframe.groupby("Trial").count() #This couns number of samples under each trials
        self.column_list = self.dataframe.drop(["Trial","Class"],axis=1) #This generate features dataframe by removing Trials and classes
        self.n_samples=n_samples #Samples taken per channel

        #assertation to assess .csv file
        assert not self.dataframe.isna().any().any(), "The CSV file contains blank values.Please modify your dataset."


    def convert_to_tensor(self)->torch.tensor:
      total=[] #To store token data as batches
      for trials in range(1,self.dataframe["Trial"].nunique()+1): #Iterare through each trials
        tokens=[] # Token stores after flattening samples under each channel
        for n_token in range(self.df_class["Class"].min()//self.n_samples): #This limits iteration regarding token to grab it only upto divislibele number samples
          flattens=[] #To store data under each flatten respect to each token
          for EEG_columns in (self.column_list.columns.tolist()): #Iterate through each channels column
              for EEG_value in range((n_token)*(self.n_samples),(n_token+1)*(self.n_samples)):#Takng n samples from each column
                  flattens.append(self.dataframe[self.dataframe["Trial"]==trials].reset_index(drop=True)[EEG_columns][EEG_value])
          tokens.append(flattens)
        total.append(tokens)
      return torch.tensor(total) #This return both tokenized data and class data

class ExcelToTensor_Class():
    """
    This is the class to tokenize the EEG signal data into required format.
    .csv file should have following format
    | Trial | Class | EEG_Channel_1 | EEG_Channel_2 | ... | EEG_Channel_N |
    -----------------------------------------------------------------------
    """
    def __init__(self,path,num_classes:int)->None:
        self.na_values =[" "]#value contain interms of regarding misiing values
        self.path = path#This is the path to csv file
        self.dataframe = pd.read_csv(path,na_values=self.na_values)#This create .csv dataframe
        self.df_class =self.dataframe.groupby("Trial").count()#This couns number of samples under each trials
        self.column_list = self.dataframe.drop(["Trial","Class"],axis=1)#This generate features dataframe by removing Trials and classes
        self.num_classes = num_classes
        assert not self.dataframe.isna().any().any(), "The CSV file contains NaN values."

    def convert_to_tensor(self)->torch.tensor:
      class_List=[]#Class list to store class of each input tensor
      for i in range(self.dataframe["Trial"].nunique()):
        for number in list(self.dataframe.groupby("Trial")["Class"])[i][1]:
          class_List.append(number-1)
          break
      class_Tensor = torch.tensor(class_List).unsqueeze(1).to(torch.int64) #Increase classes dimention by 1 to later use for trainingpurposes of transfomer
      output_class = torch.zeros(self.dataframe.groupby("Trial").count().shape[0],self.num_classes)
      output_class = output_class.scatter_(1, class_Tensor, 1)
      return output_class

class PositionEncoding(nn.Module):
    def __init__(self, input_seq, d_model):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, input_seq, d_model))

    def forward(self, x):
        x = x + self.position_embedding
        return x

class LayerNormalization(nn.Module):
    """
    This to class to define layer normalization block in transfomer archtecture
     """

    def __init__(self, features: int, eps:float=10**-6):
        super().__init__()
        self.eps = eps # To address division by zero error
        self.gamma = nn.Parameter(torch.ones(features)) #Use this as a trainable parameter for the model
        self.beta = nn.Parameter(torch.zeros(features)) #Use this as a trainable parameter for the model

    def forward(self, x):
        # x: (batch, token_len, hidden_size)
        mean = x.mean(dim = -1, keepdim = True) # (batch, token_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, token_len, 1)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta #Reffered to the paper output

class FeedForwardBlock(nn.Module):

    """
    This class defines transfomer feedfoward layer. It deviates from general transfomer artitecthure by using GELU activation
    """

    def __init__(self, v_dimentions: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(v_dimentions, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, v_dimentions) # w2 and b2

    def forward(self, x):
        # (batch, token_len, v_dimentions) --> (batch, token_len, d_ff) --> (batch, token_len, v_dimentions)
        return self.linear_2(self.dropout(torch.nn.functional.gelu(self.linear_1(x.to(torch.float32)))))

class MultiHeadAttentionBlock(nn.Module):

    """
    This is class to define multihead attention block for the transfomer

    """

    def __init__(self, v_dimentions: int, n_heads: int, dropout: float):
        super().__init__()
        self.v_dimentions = v_dimentions # Embedding vector size
        self.heads = n_heads # Number of heads
        # Make sure v_dimentions is divisible by heads
        assert v_dimentions % n_heads == 0, "v_dimentions is not divisible by number of heads"

        self.d_k = v_dimentions // n_heads # Dimension of vector seen by each head
        self.w_q = nn.Linear(v_dimentions, v_dimentions) # Weight matrix initiating for query
        self.w_k = nn.Linear(v_dimentions, v_dimentions) # Weight matrix initiating for key
        self.w_v = nn.Linear(v_dimentions, v_dimentions) # Weight matrix initiating for values
        self.w_o = nn.Linear(v_dimentions, v_dimentions) # Weight matrix initiating for heading values
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, token_len, d_k) --> (batch, h, token_len, token_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, token_len, token_len) # Apply softmax along last dimention
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, token_len, token_len) --> (batch, h, token_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, token_len, v_dimentions) --> (batch, token_len, v_dimentions)
        key = self.w_k(k) # (batch, token_len, v_dimentions) --> (batch, token_len, v_dimentions)
        value = self.w_v(v) # (batch, token_len, v_dimentions) --> (batch, token_len, v_dimentions)

        # (batch, token_len, v_dimentions) --> (batch, token_len, n_heads, ) --> (batch, n_heads, token_len, d_k)
        # Final matrix should be addressing sequnce for each heads .view and .transpose used
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, n_heads, token_len, d_k) --> (batch, token_len, n_heads, d_k) --> (batch, token_len, v_dimentions)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        # (batch, token_len, v_dimentions) --> (batch, token_len, v_dimentions)
        return self.w_o(x)

class ResidualConnection(nn.Module):

        """
        This is the class to objectify the Residual connection between inputs and Normalization layers
        """

        def __init__(self, features: int, dropout: float):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x))) #When consider this layer even artitecthure says sublayer first normalization
                                                            #normalization after that coding should be carried out like this
class EncoderBlock(nn.Module):
    """
    This is the aggregation of elemnts of class which createsd to build the encorder part
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block #Create an object using multiheattention class
        self.feed_forward_block = feed_forward_block #Create an object using feed forward class
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) #Create residual connections list

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # This defnes the first residual connection between input and multihead attention layer
        x = self.residual_connections[1](x, self.feed_forward_block) #Then residual connections has been applied to the second residual connections
        return x

class Encoder(nn.Module):
    """"
    Since encoder consists of many encoder this blocks  defines the full encoder object
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers #Input  list of layers into the encoder
        self.norm = LayerNormalization(features) #Define the layer normalization

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask) #apply layer by layer functioning to inputs
        return self.norm(x) #Normalize final output

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class ProjectionLayer(nn.Module):
    """
    This is the projection layer to convert (N_tokens X v_dimensions) ------> (1 X v_dimensions) -------> (1 X N_class)
    """

    def __init__(self, d_model: int, N_class: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, N_class, bias=False)  # Linear layer for matrix conversions

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        x = x.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        x = x.squeeze(1)  # Remove the singleton dimension
        return torch.softmax(self.proj(x), dim=-1)  # (batch, N_class)

class Projector(nn.Module):
    """
    This is the projection layer to convert (N_tokens X v_dimensions) ------> (1 X v_dimensions) -------> (1 X N_class)
    """

    def __init__(self, d_model: int, features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, features, bias=False)  # Linear layer for matrix conversions

    def forward(self, x) -> None:
        x = x.squeeze(1)  # Remove the singleton dimension
        x = self.proj(x)
        return x

class Average_Encorder_Block(nn.Module):

  """
  This is the class to process decoder output of Low level transformer into a form where
  it is able to input into high level transformer
  """
  def __init__(self,dropout:float)->None:
      super().__init__()
      self.dropout = nn.Dropout(dropout)

  def forward(self, x)->torch.tensor:
      x=x.mean(dim=-2,keepdim = True) #This is the part which averages through token stack length
      return self.dropout(x) #Add dropout to stop overfitting

class Low_LeveL_Transformer(nn.Module):

    """This is the transformer complete model by combining modulate classes of full transformer,Since this is a customizeed to use with EEG signal classification
    decoder part has been removed.Further advanced may be caused tio add decoder to the transformer."""

    def __init__(self,encoder_block:EncoderBlock,projection:ProjectionLayer,average_block:Average_Encorder_Block(0.1),features:int,n_encoder_layers:int,mode:bool=False): # type: ignore
        super().__init__()
        encoder_List =[] #This is the list to store  the complete encoder layers
        self.features= features #This is the features regarding feedfoward layer in encoder block
        self.n_encoder_layer = n_encoder_layers #This is the number of encoder layers use in transfomer artitecture
        self.mode = mode
        for full_encoder in range(n_encoder_layers): #This is the iteration  through encoder layers to add encoder blocks
          self.encoder_block = encoder_block
          encoder_List.append(self.encoder_block)
        self.encoder = Encoder(self.features,nn.ModuleList(encoder_List))
        self.encoder_full_list=nn.ModuleList(encoder_List)#Conertion to layers for the module list
        self.projection = projection #Projection layer of the transformer only one projection layer has been used with respect to final output
        self.average_block = average_block
        self.norm = LayerNormalization(self.features) #Define the layer normalization

    def forward(self, x, src_mask=None):
      x = self.encoder(x,src_mask)
      projection_output = self.projection(x) #Final output  has  been projected to 1XNclass dimention vector for the prediction
      return projection_output