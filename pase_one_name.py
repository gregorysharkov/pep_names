from synonym_finder.synonyms_finder import SynonymsFinder
from synonym_finder.synonyms_finder_settings import GLOBAL_SETTINGS

if __name__=="__main__":
    #parse name
    name = "Stéphane"
    finder = SynonymsFinder(name,GLOBAL_SETTINGS)
    finder.fit()
    print(finder.collect_labels())