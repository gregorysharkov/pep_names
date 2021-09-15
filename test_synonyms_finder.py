from time import time
from threading import BoundedSemaphore
from synonyms_finder import Synonyms_finder

def timeit(method):
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
        req = Synonyms_finder(name)
        req.start()
        threads.append(req)

    return_dict = {}
    for res in threads:
        res.join()
        return_dict.update(res.synonyms_dict)
    return return_dict

@timeit
def test_one_name(*args,**kwargs):
    name = "julia"
    syn = Synonyms_finder(name)
    syn.fit()
    print(syn)

@timeit
def test_several_names():
    names = ["Bill","George","Michael","John","Gregory"]
    maximumNumberOfThreads = 5
    threadLimiter = BoundedSemaphore(maximumNumberOfThreads)
    names_dict = parse_names(names,threadLimiter)
    print(f"names_dict contains {len(names_dict)} elements.")

if __name__ == '__main__':
    # test_one_name()
    test_several_names()