from stream import DataProcessor
from threading import Condition
import time
import pandas as pd
from model import Low_LeveL_Transformer, EncoderBlock, MultiHeadAttentionBlock, FeedForwardBlock, ProjectionLayer, Average_Encorder_Block
import torch
from gui import Worker, GUI
from PyQt5 import QtCore, QtWidgets
import sys

# Load EEG Data
# file = "Session1_Sub4_Class2_MI_100LP_Resampled250.csv"
# data = pd.read_csv(file, usecols=range(2,32)).to_numpy()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Parameters regarding full model
num_samples = 4
num_channel = 30
num_classes = 4
num_heads = 2
num_encoder_layers = 4
drop_out_rate = 0.4
learning_rate = 0.0001
token_dimentions = num_channel*num_samples
features = num_channel*num_samples
path=""
mode=False

model = Low_LeveL_Transformer(EncoderBlock(token_dimentions,MultiHeadAttentionBlock(token_dimentions,num_heads,drop_out_rate),FeedForwardBlock(token_dimentions,token_dimentions,drop_out_rate),drop_out_rate),ProjectionLayer(token_dimentions,num_classes),Average_Encorder_Block(drop_out_rate),features=token_dimentions,n_encoder_layers=num_encoder_layers,mode=mode).to(device)

# Loading the Model
file_path_1= 'LLT_15_1_93'#Last digit to be used in the training loop
model_state_dict = torch.load(file_path_1, map_location=torch.device(device))
model.load_state_dict(model_state_dict)
model.eval()

model = DataProcessor(model)

app = QtWidgets.QApplication(sys.argv)
mainWindow = GUI(model)
mainWindow.show()
sys.exit(app.exec_())