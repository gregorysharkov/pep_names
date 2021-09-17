import json
import urllib
from dataclasses import dataclass
from synonyms_finder_utils import fetch_url


#https://www.wikidata.org/w/api.php?action=wbgetentities 
#   &format=json
#   &sites=enwiki
#   &titles=Julia
#   &props=info%7Cclaims
#   &normalize=1

@dataclass
class ClaimParam():
    site:str
    title:str
    base_url:str="https://www.wikidata.org/w/api.php"

    def encode(self):
        return f"{self.base_url}?{urllib.parse.urlencode(self._get_param())}"

    def _get_param(self):
        return {
            "action":"wbgetentities",
            "sites":self.site,
            "titles":self.title,
            "format":"json",
            "props":"info|claims|labels",
            "normalize":1,
        }


class WikiDataItem():
    """
    The class is responsible for retrieving the right information about each wiki data result
    we will be working with:
    * id of the item
    * connections (claims) to other units. Stored in the form of dictionary {type_of_connection:id}
    * different ways to write it
    """
    def __init__(self,response):
        self.entities = None
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
        return False

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
class SearchGrid():
    name:str
    sites = ["enwiki","eswiki","ruwiki"]
    results = []

    def fit(self):
        for site in self.sites:
            url = ClaimParam(site,self.name).encode()
            result = WikiDataItem(json.load(fetch_url(url)))
            if result not in self.results:
                self.results.append(result)

        for res,i in zip(self.results,range(len(self.results))):
            print("***********************************")
            print(f"Result #{i+1}:\n{repr(WikiDataItem(res))}")


def main():
    grid = SearchGrid("Julia")
    grid.fit()

if __name__ == '__main__':
    main()