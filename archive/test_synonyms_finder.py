from time import time
from threading import BoundedSemaphore
#from synonyms_finder import Synonyms_finder
from synonyms_finder_refacto import SynonymsFinder
from synonyms_finder_settings import GLOBAL_SETTINGS

def timeit(method):
    """decorator function to measure execution time"""
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print(f"Execution time: {(te - ts):2.2f} sec")
        return result

    return timed

def parse_names(names,threadLimiter=None):
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
        return_dict.update(res.collect_labels())
    return return_dict

@timeit
def test_one_name():
    name = "Julia"
    syn = SynonymsFinder(name,GLOBAL_SETTINGS)
    syn.fit()
    print(syn.collect_labels())

@timeit
def test_several_names():
    names = ["Bill","George","Michael","John","Gregory"]
    maximumNumberOfThreads = 5
    threadLimiter = BoundedSemaphore(maximumNumberOfThreads)
    names_dict = parse_names(names,threadLimiter)
    print(names_dict)

if __name__ == '__main__':
    # test_one_name()
    test_several_names()