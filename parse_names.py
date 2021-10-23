import string
import json
from utils.utils import load_data
from functools import reduce
from threading import BoundedSemaphore
from synonym_finder.synonym_finder import SynonymsFinder
from synonym_finder.synonyms_finder_settings import GLOBAL_SETTINGS
from utils.proxy_handler import SgProxyHandler

def collect_names(lst: list[str]) -> list[str]:
    '''Function collects unique names from a given list of strings'''
    #concatenate all lists
    return_list = reduce(lambda x,y: x+y,lst)
    return sorted(set(return_list))

def parse_list(names,threadLimiter=None):
    '''Function parses a list of names. Each name is parsed in a separate thread'''
    threads = []
    for name in names:
        req = SynonymsFinder(name,GLOBAL_SETTINGS,threadLimiter=threadLimiter)
        req.start()
        threads.append(req)

    return_dict = {}
    for res in threads:
        res.join()
        return_dict.update(res.collect_labels())
    return return_dict

def parse_names(names):
    """Function parces a list of names with SynonymFinder"""
    maximumNumberOfThreads = 40
    threadLimiter = BoundedSemaphore(maximumNumberOfThreads)
    names_dict = parse_list(names,threadLimiter)
    return names_dict

def main():
    #set up proxy handler
    credentials_path = "conf\\local\\credentials.yml"
    sg_proxy = SgProxyHandler(credentials_path)
    sg_proxy.setup_handler()

    #read data
    path_to_data = "data\\raw\\bg_names.csv"
    raw_data = load_data(path_to_data,"bg_name")
    unique_names = collect_names(raw_data.bg_name_split)
    
    #parse unique names
    names_dict = parse_names(unique_names)
    print(names_dict)
    
    str_names_dict = json.dumps(obj=names_dict,indent=2,ensure_ascii=True)
    with open("data\\dict\\dictionary_en_fr_ar_ru.yml", "w") as outfile:
        outfile.write(str_names_dict)
        outfile.close()
    pass

if __name__ == "__main__":
    main()