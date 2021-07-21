import pandas as pd

def load_data(path,debug=True):
        '''Function loads data from the given path
        '''
        data = pd.read_csv(path,sep=",")
        data = data[["governor"]].drop_duplicates()
        data["governor_split"] = data.governor.str.split(" ")
        return data
