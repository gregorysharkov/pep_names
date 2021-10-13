import re
import json
import urllib
from threading import Thread, BoundedSemaphore
from itertools import chain
from functools import reduce
from dataclasses import dataclass, field
from synonyms_finder_utils import fetch_url, IdRequestParam,TitleRequestParam
from synonyms_finder_settings import GLOBAL_SETTINGS


class WikiDataItem():
    """
    The class is responsible for retrieving the right information about each wiki data result
    we will be working with:
    * id of the item
    * connections (claims) to other units. Stored in the form of dictionary {type_of_connection:id}
    * different ways to write it
    """
    def __init__(self,response):
        self.id = None
        self.claims = None
        self.labels = None
        
        self.response = response
        #first let's check if our resonse has any entities, if yes we will need to store the id
        if "entities" in response:
            self.entities_raw = response["entities"]
            #working with ids
            self.ids = [id for id in self.entities_raw]
            self.id = self.ids[0]
            if len(self.ids) > 1:
                print("Warning. Result returned many ids. Only the first one will be kept")
        
        #now we need to collect claims properly. We need only the vlaim value and the list of associated ids
            if "claims" in self.entities_raw[self.id]:
                self.claims_raw = self.entities_raw[self.id]["claims"]
                self.claims = {}
                for claim,claim_value_raw in self.claims_raw.items():
                    self.claims.update({claim:self._collect_connections(claim_value_raw)})

        #one last thing we will do is collect all available labels
            if "labels" in self.entities_raw[self.id]:
                self.labels_raw = self.entities_raw[self.id]["labels"]
                self.labels = self._collect_labels(self.labels_raw)

    def __repr__(self):
        return_string = f'{self.id=}\n{self.labels=}'#\n{self.claims=}
        return return_string

    def __eq__(self,other):
        #if the id of two wikidata entities are the same, we can consider items to be the same
        if isinstance(other,WikiDataItem):
            return self.id == other.id

        if (other is None) and (self.id == -1):
            return True

        return False

    def check_instance_type(self,property,instance_list)->bool:
        """Function checks if the given instance has at least one property from the instance_list"""
        if (set(self.claims[property])&set(instance_list)):
            return True
        return False

    def get_property(self,property):
        try:
            return self.claims[property]
        except KeyError:
            return None

    def _collect_connections(self,data):
        """function collects the nature of the connection (associated with P...) and the corresponding IDs
        Some properties do not have an id inside the value. For example P3878 which is Soundex of the search item.
        Another example is P1705, which stands for native label In this case we will them as keys in the dictionary
        without taking their values.
        we will keep only connections to wikidata items
        """
        return_list = []
        for i in range(len(data)):
            try:
                if "Id" in [str(x).capitalize() for x in data[i]["mainsnak"]["datavalue"]["value"]]:
                    return_list.append(data[i]["mainsnak"]["datavalue"]["value"]["id"])
            except KeyError:
                pass
        return return_list

    def _collect_labels(self,data):
        """function collects unique labels from the labels dictionary"""
        return_list = []
        for _lang,value in data.items():
            return_list.append(value["value"])
        return sorted(set(return_list))


@dataclass
class WikiItemsGroup():
    """
    This class is responsible for retrieving requests from different
    sites and manipulating results (keeping only unique results, providing ids, and their properties)
    """
    name:str = None
    id:str = None
    sites: list[str] = field(default_factory=list)
    results: list[WikiDataItem] = field(default_factory=list)

    def __add__(self,other):
        '''
        function handles combination of two wikiitems group
        we need this function to handle cases when we need to combine two
        groups together
        '''
        # first, let's combine ids and names
        self.name = self._combine_str_values(self.name,other.name)
        self.id = self._combine_str_values(self.id,other.id)
        
        # now, let's combine results. We need to keep only unique WikiItmes in results
        for item in other.results:
            if item not in self.results:
                self.results.append(item)

        return self

    def _combine_str_values(self,value_a,value_b):
        '''utility function to combine two values'''
        if (value_a is None) & (value_b is None):
            value_a = None
        elif value_b is None:
            value_a = value_a
        elif value_a is None:
            value_a = value_b
        else:
            if isinstance(value_a,list) & isinstance(value_b,list):
                value_a = sorted(set(value_a + value_b))
            elif isinstance(value_a,str) & isinstance(value_b,list):
                value_a = sorted(set([value_a]+value_b))
            elif isinstance(value_a,list) & isinstance(value_b,str):
                value_a = sorted(set(value_a + [value_b]))
            else:
                value_a = sorted(set([value_a,value_b]))

        return value_a
        
    def __repr__(self):
        return_string = f"\n{self.name=},{self.id=}\nResults:\n" 
        for result in self.results:
            return_string = return_string + f"\n{result=}\n"

        return return_string

    def fit(self):
        '''Collect information about the name from the list of sites'''
        for site in self.sites:
            #prepare a proper url based on the available data
            if self.name is None:
                param = IdRequestParam(site,self.id)
            else:
                param = TitleRequestParam(site,self.name)
            url = param.encode()

            #fetch WikiDataItem and add it to results
            result = WikiDataItem(json.load(fetch_url(url)))
            if result not in self.results:
                self.results.append(result)
        
        return self

    def get_ids(self):
        '''get all top level ids from the group'''
        return [res.id for res in self.results]

    def get_property(self,property):
        '''get common list of claims with the given property'''
        properties = []
        for el in self.results:
            claim = el.claims.get(property)
            if claim:
                properties.append(claim)

        return sorted(set(chain(*properties)))

    def collect_labels(self):
        """Function combines all labels of all items of the group into one list"""
        labels = [label for element in self.results for label in element.labels]
        return sorted(set(labels))



class SynonymsFinder(Thread):
    '''
    The class is responsible for finding synonyms of a given name
    General logic of the fit step
    1. First we create a group of items related to the request
    2. items that are instances of NAME instances have to be treaded as names
        * serach for list of items with property P460 (said to be the same)
        * get labels of the parent instance and the related instances
    3. items that instances of disambiguation page should be treaded separately
        * check contents of the disambiguation page
        * if any of these links is a name instance, treat it process it like a name
    '''
    sites: list[str] = None
    items:WikiItemsGroup = None

    def __init__(self,name,global_settings,
                 group=None, target=None, 
                 threadLimiter=None, args=(), kwargs=()):
        super(SynonymsFinder,self).__init__(group=group,target=target,name=name)

        self.args = args
        self.kwargs = kwargs

        self.thread_limiter = threadLimiter

        self.name = name
        self.global_settings = global_settings
        self.items = None

        self.sites = self.global_settings["SITES"]
        self.name_instances = self.global_settings["NAME_INSTANCES"]
        self.disambiguation_instances = self.global_settings["DISAMBIGUATION_INSTANCES"]

        self.level = 0

    def __repr__(self):
        return_string = f"Global_name: {self.name}\nResults:\n"
        if self.items is None:
            return_string = return_string + "—"
        else:
            for item in self.items.results:
                return_string = return_string + repr(item)
        return return_string

    def fit(self):
        # create the top-level group for the requested name
        group = WikiItemsGroup(name=self.name, sites=self.sites)
        group.fit()
        group = group + self._collect_children(group,group.id)
        self.items = group

    def collect_labels(self):
        '''Interface to collect labels from the group'''
        if self.items:
            return_values = self.items.collect_labels()
        else:
            return_values = None

        # Final touch: some values may contain some noise, like "/" or (values in brakets)
        # we need to take kare of that
        for el in return_values:
            if "/" in el:
                new_split = el.split(r" / ")
                return_values.remove(el)
                for new_el in new_split:
                    return_values.append(new_el)
            #it seems like another encoding
            elif "/" in el:
                new_split = el.split(r"/")
                return_values.remove(el)
                for new_el in new_split:
                    return_values.append(new_el)

        
        return_values = sorted(set(re.sub(r" \(.+?\) ?","",x) for x in return_values))
        return {self.name:return_values}

    def _collect_children(self,group:WikiItemsGroup,ids:list[str]=None) -> WikiItemsGroup:
        '''Recurcive function to collect all children of a given group'''
        new_ids = self._check_for_new_ids(group,ids)

        #print(f"Ids captured ad level {self.level}: {new_ids}")
        self.level += 1

        ids = [] if not ids else ids
        if new_ids:
            child_group = self._create_group_from_ids(new_ids)
            return_group = group + self._collect_children(child_group,ids+new_ids)
        else:
            return_group = group

        return return_group

    def _check_for_new_ids(self,group,ids:list[str]=None):
        '''Function checks if a group contains any new ids compared to ids of the group'''
        if ids:
            child_ids = self._get_child_ids(group)
            return_list = sorted(x for x in child_ids if x not in ids)
        else:  
            return_list = self._get_child_ids(group)
        return return_list

    def _get_child_ids(self,group: WikiItemsGroup) -> list[str]:
        '''Collect list of name synonym ids'''
        if group.results:
            if (group.get_property("P460")):
                children = group.get_property("P460")
        else:
            children = []

        return children

    def _create_group_from_ids(self,ids: list[str])->WikiDataItem:
        '''Create a new WikiItemsGroup from a list of ids'''
        fetched_items = [WikiItemsGroup(id=x,sites=self.sites).fit() for x in ids]
        combined_group = reduce(lambda x,y: x+y, fetched_items)
        return combined_group

    def run(self):
        """function that is called whenever the thread is started"""
        if self.thread_limiter:
            self.thread_limiter.acquire()
        
        try:
            self.fit()
        finally:
            if self.thread_limiter:
                self.thread_limiter.release()
            print(f"Done with {self.name}")

from time import time
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


def main():
    test_one_name()
    #test_several_names()

if __name__ == '__main__':
    main()