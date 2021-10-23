import re
import json
from pandas.io import parsers
from time import sleep
from threading import BoundedSemaphore
import pandas as pd
from utils import load_data
from synonyms_finder_refacto import SynonymsFinder
from synonyms_finder_settings import GLOBAL_SETTINGS
#from synonyms_finder import Synonyms_finder



def get_unique_names(original, col):
        '''gets unique values of a given column that are in all sub elements of this column'''
        return_seq = sorted(original[[col]].explode(col).drop_duplicates()[col])
        return return_seq


def parse_names(names,threadLimiter):
        '''Function parses a list of names. Each name is parsed in a separate thread
        '''
        threads = []
        for name in names:
                req = SynonymsFinder(name,GLOBAL_SETTINGS,threadLimiter=threadLimiter)
                req.start()
                threads.append(req)

        return_dict = {}
        for res in threads:
                res.join()
                return_dict.update(res.synonyms_dict)

        return return_dict
        

if __name__ == "__main__":
        maximumNumberOfThreads = 40
        threadLimiter = BoundedSemaphore(maximumNumberOfThreads)

        original = load_data("data\\source\\united_states_governors.csv")

        unique_names = get_unique_names(original,"governor_split")
        print(len(unique_names))

        names_dict = parse_names(unique_names,threadLimiter)
        with open("data\\dict\\names.json", "w",encoding="utf-8") as file:
                file.write(json.dumps(names_dict,indent=2))

        # name = "julia"
        # syn = Synonyms_finder(name)
        # syn.fit()
        # print(syn)
