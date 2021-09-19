import json
import urllib
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
        return_string = f'{self.id=}\n{self.claims=}\n{self.labels=}'
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
            if "id" in data[i]["mainsnak"]["datavalue"]["value"]:
                return_list.append(data[i]["mainsnak"]["datavalue"]["value"]["id"])
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
        '''function handles combination of two wikiitems group'''
        # first, let's combine ids and names
        if (self.name is None) and (other.name is None):
            self.name = None
        elif self.name is None:
            self.name = other.name
        else:
            self.name = [self.name, other.name]

        if (self.id is None) & (other.id is None):
            self.id = None
        elif self.id is None:
            self.id = other.id
        else:
            self.id = [self.id, other.id]

        # now, let's combine results. We need to keep only unique WikiItmes in results
        for item in other.results:
            if item not in self.results:
                print(f"adding element {item.id}")
                self.results.append(item)
            # else:
            #     print(f"ignoring element {item.id} because it is already in the list")
        return self

    def __repr__(self):
        return_string = f"{self.name=},{self.id=}\n{self.sites=}\nResults:\n"
        for result in self.results:
            return_string = return_string + f"{result=}\n"

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
            try:
                properties.append(el.claims[property])
            except KeyError:
                continue
        return sorted(set(chain(*properties)))


@dataclass
class SynonymsFinder():
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
    name:str
    global_settings : dict
    sites: list[str] = field(default_factory=list)
    items:WikiItemsGroup = field(default_factory=list)

    def __post_init__(self):
        self.sites = self.global_settings["SITES"]
        self.name_instances = self.global_settings["NAME_INSTANCES"]
        self.disambiguation_instances = self.global_settings["DISAMBIGUATION_INSTANCES"]

    def fit(self):
        group = WikiItemsGroup(name=self.name, sites=self.sites)
        group.fit()
        print(f"Fitting {self.name}")
        print(f"The main group has the following results:\n{group.results}")
        self.items = group.results
        return self

    def fetch_children(self):
        '''
        function creates 2 list WikiItemsGroups for children:
        * one list of wiki groups for each synonym instance
        * one list of wiki groups for each disambiguation instance
        '''
        for el in self.items:
            self._process_name_instance(el)
            # self._process_disambiguation_page(el)


    def _process_name_instance(self,element:WikiDataItem):
        '''function returns a group of wikiitems that are mentioned to be the same as the given element'''
        print(f"    Processing synonyms of {self.name=}")
        #first let's check that this is a name instance if not, none is returned
        if not element.check_instance_type("P31",self.name_instances):
            return None

        #next let's check that the item has P460 property meaning that it has synonyms        
        name_synonyms = element.get_property("P460")
        if name_synonyms is None:
            return None
        print(f"Synonyms that have to be checked: {name_synonyms}")
        fetched_synonyms = [WikiItemsGroup(id=id,sites=self.sites).fit() for id in name_synonyms]

        combined_group = reduce(lambda x,y: x+y, fetched_synonyms)
        print(combined_group)

        return fetched_synonyms

    def _process_disambiguation_page(self,element:WikiDataItem):
        check = element.check_instance_type("P31",self.disambiguation_instances)
        if not check:
            return None

        print(f"Checking if element {element.id} is a disambiguation page: {check}")
        pass


def main():
    finder = SynonymsFinder("Grigory",GLOBAL_SETTINGS)
    finder.fit()
    finder.fetch_children()
    # group = WikiItemsGroup(name="Julia",sites=["enwiki","frwiki"])
    # group.fit()
    # print(group)

    # print(group.get_property("P1560"))
    

if __name__ == '__main__':
    main()