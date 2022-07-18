from data.CustomdataSet import CustomDataset
from torch.utils.data import DataLoader
import os

def main():
    path= os.getcwd()+'\data\\testData1.csv'
    print(path)
    data=CustomDataset(path)
    print(len(data))
    print(data[:-1])
    print("____________________________________________________________________")

    dataloader = DataLoader(data, batch_size=64, shuffle=True)
    #for i, batch in enumerate(dataloader):
        #print(i, batch)

if __name__ == '__main__':
    main()