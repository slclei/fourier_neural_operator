import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image
import csv

#custom a simple data set
#read data from ../data/fileName
#data type is: 1st line: x1,x2,y; other lines: data for x1, x2, y
#testData1 fomular: y=x1+x2
class CustomDataset(Dataset):
    def __init__(self, fileName):
        #label records 1st line, i.e. x1,x2,y for testData1
        self.label=[]
        #x records input data, i.e. data of x1, x2 for testData1
        self.x=[]
        #y records output data, i.e. data of y for testData1
        self.y=[]
        self.len=0

        #read data into list of strings
        with open(fileName,encoding="UTF-8-sig") as csv_file:
            csv_reader=csv.reader(csv_file,delimiter=',')
        
            for data in csv_reader:
                #record label in 1st line
                if self.len==0:
                    self.label.append(data)
                #record x data and y data (last col)
                else:
                    self.x.append(data[:-1])
                    self.y.append(data[-1])
                #increment length
                self.len+=1

    def __len__(self):
        #return length of data
        return self.len-1

    def __getitem__(self, idx):
        #return x and y data at index 'idx'
        return self.label,self.x[idx],self.y[idx]