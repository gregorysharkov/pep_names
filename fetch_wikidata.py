import re
from typing import Match
import urllib
import json
from pandas.io import parsers
import requests
from threading import *
from time import *
import time
import pandas as pd
from utils import load_data

def fetch_url(url):
        current_delay=.1 #set the initial retry delay to 100 ms
        max_delay=10     #set maximum retry delay to 10 seconds
        while True:
                try:
                        response = urllib.request.urlopen(url)
                except urllib.error.URLError:
                        print("URL error. Falling to the retry loop with: "+url)
                        pass
                except urllib.error.HTTPError:
                        print("HTTP error. Falling to the retry loop with: "+url)
                        pass 
                else:
                        #if there is no error, we can return the response
                        return response

                #skip this result after mutliple fails:
                if current_delay > max_delay:
                        print("Too many fails, going to the next one after 60 seconds")
                        sleep(60)
                        return None

                #add some wait time after the recent fail
                print("Waiting", current_delay, "seconds before retrying url: ",url)
                sleep(current_delay)
                current_delay *=2


class Synonyms_finder(Thread):
    def __init__(self,request,group=None, target=None, name=None,
                 args=(), kwargs=(), verboise=None):
        super(Synonyms_finder,self).__init__(group=group,target=target,
                                             name=name)
        self.args = args
        self.kwargs = kwargs
        self.req = request
        self.__load_credentials()
        self.__install_openner()
        
        #url used to fetch results
        self.url = "https://www.wikidata.org/w/api.php"
        #list of sites that we will be looking for
        self.sites = ["enwiki","frwiki","eswiki","ruwiki","chwiki"]

        #instances of classes we want to filter these are all types of name instances
        #       * family name (Q101352)
        #       * given name (Q202444)
        #       * unisex given name (Q3409032)
        #       * female given name (Q11879590)
        #       * male given name (Q12308941)
        #       * double name (Q1243157)
        #       * matronymic (Q1076664)
        #       * patronymic (Q110874)
        #       * disambiquation page (Q4167410)
        self.instances = ["Q101352","Q202444","Q3409032","Q11879590","Q12308941","Q1243157","Q1076664","Q110874","Q4167410"]
        self.qs = [] #list of qs associated with the value
        self.alias = {}
        self.synonyms = []
        self.synonyms_dict = {}

    def fit(self):
        '''Function finds all synonyms of term stored in self.req'''

        self.qs = self.__get_qs_norm(self.req)
        for q in self.qs:
            self.alias.update(self.__get_alias(q))

        #now, let's fetch labels for linked elements. Only elements with propery "said to be the same as" will be kept
        labels_dict = {}
        for key,value in self.alias.items():
            if value:
                self.alias.update(self.__get_labels(key))
                for alias in value:
                        labels_dict.update(self.__get_labels(alias))

        combined_list = []
        for _key, value in labels_dict.items():
                combined_list = [*combined_list,*value]

        self.synonyms = list(set(combined_list))
        self.synonyms_dict = {self.req:self.synonyms}

    def run(self):
        """override parent method, that is called when the thread is started
        """
        threadLimiter.acquire()
        try:
            self.fit()
        finally:
            threadLimiter.release()
            print(f"Done with: {self.req}")

    def __str__(self):
        '''override standard method for printing
        '''
        s = f"Request:\t{self.req}\n"+\
                f"Qs:\t\t{self.qs}\n"+\
                f"Alias:\t\t{self.alias}\n"+\
                f"Synonyms:\t{self.synonyms}\n"
        return s

    def __load_credentials(self):
        '''Load proxy credentials
        use it if you have proxy
        '''
        # credentials = open(".\conf\local\credentials.txt","r")
        # self.key = credentials.readline()[:-1]
        # self.user = credentials.readline()[:-1]
        # self.pwd = credentials.readline()[:-1]
        # credentials.close()

    def __install_openner(self):
        '''Create a proxy gate for requests
        use it if you have proxy
        '''
        # proxy_url = "@proxy-sgt.si.socgen:8080"
        # proxy = {"https": "https://" + self.user + ":" + self.pwd + proxy_url}
        # proxy_support = urllib.request.ProxyHandler(proxy)
        # opener = urllib.request.build_opener(proxy_support)
        # urllib.request.install_opener(opener)

    def __get_qs_norm(self,text):
        '''Function fetches all qs associated with the given string in the given sites

        we have two options:
        1. either the q that we get is already of a desired type, then we just add it
        2. or the q that we get is a disambiguation page. In this case we will need to check if this 
        page points toward one of Qs that we are interested in
        
        Args:
                string: string to be searched for
                url: url used inthe search query
                sites: wiki sites to be used
        
        Returns:
                list of Q....
        '''
        for site in self.sites:
                params = urllib.parse.urlencode({
                                "action" : "wbgetentities",
                                "sites" : site,
                                "titles": text,
                                "props" : "claims",
                                "normalize" : 1,
                                "format": "json"
                },)

                url = f"{self.url}?{params}"
                response = fetch_url(url)
                result = json.load(response)
                #check if we have a match
                if "entities" in result:
                        for el in result["entities"]:
                                if "Q" in el:
                                        #now, let's check that our element has a proper class (property P31):
                                        if self.__check_p31(result,el):
                                                self.qs.append(el)

                                        #now, let's check if our element has a disambiguation page
                                        if self.__check_disambiguation_page(result,el):
                                                self.__fetch_qs_from_disambiguation_page(self,el)
        
        #once we are done, let's get rid of all duplicates in our list
        self.qs = list(set(self.qs))

        return self.qs

    def __check_p31(self,result,q):
        '''Function checks if a given q has a property "instance of" and this property
        belongs to list of classes we want

        Args:
                result: json list containing output from wikidata
                q: q that we are looking for

        Returns:
                a boolean indicating whether the given q has a propery from one of the self.instances
        '''
        whish_list = self.instances
        entity = result["entities"][q]
        match = False
        if "P31" in entity["claims"]:
                for instance in entity["claims"]["P31"]:
                        if instance["mainsnak"]["datavalue"]["value"]["id"] in whish_list:
                                match = True
        return match

    def __check_disambiguation_page(self,result,q):
        '''function checks if a provided q is a disambiguation page

        Args:
                result: json list containing output from wikidata
                q: q that we are looking for

        Returns:
                a boolean indicatin whetehr the given q has is a disambiguation page
        '''
        match = False
        entity = result["entities"][q]

        if "P31" in entity["claims"]:
                for instance in entity["claims"]["P31"]:
                        if instance["mainsnak"]["datavalue"]["value"]["id"] == "Q4167410":
                                match = True

        return match

    def __fetch_qs_from_disambiguation_page(self,result,q):
        '''function checks the disambiguation page, searches if it points to any name (member of self.instances)
        if yes, it adds all name instances into the list

        Args:
                q: Q of the disambiguation page
        Returns:
        '''
        params = urllib.parse.urlencode({
                        "action" : "wbgetentities",
                        "sites" : "enwiki",
                        "ids": q,
                        "props" : "claims",
                        "normalize" : 1,
                        "format": "json"
        },)

        url = f"{self.url}?{params}"
        response = fetch_url(url)
        result = json.load(response)

        #if the result is not empty
        if "entities" in result:
                #let's check claims that P1889 is among the claims
                #P1889 stands for "different from"
                entity = result["entities"][q]
                for claim_id, claim_value in entity["claims"].items():
                        if claim_id == "P1889":
                                for el in claim_value:
                                        #for every claim, we need to check its status.
                                        #wheter it is in our wishlist
                                        _id = el["mainsnak"]["datavalue"]["value"]["id"]

                                        #in order to save requests, let's check if this _id has already been added
                                        if _id in self.qs:
                                               pass
                                        else: 
                                                _params = urllib.parse.urlencode({
                                                        "action" : "wbgetentities",
                                                        "sites" : "enwiki",
                                                        "ids": _id,
                                                        "props" : "claims",
                                                        "normalize" : 1,
                                                        "format": "json"
                                                })
                                                _result = json.load(fetch_url(f"{self.url}?{_params}"))
                                                #if the type is good, we keep it
                                                if self.__check_p31(_result,_id):
                                                        self.qs.append(_id)

    def __get_alias(self,q):
        '''Function gets alias names for qs
        Args:
                qs: list of qs to be searched
        Returns:
                list of names located in P460
        '''
        params = urllib.parse.urlencode({
                "action":"wbgetentities",
                "ids":q,
                "props":"info|claims",
                "format":"json"
        })

        url = f"{self.url}?{params}"
        response = fetch_url(url)
        alias_res = json.load(response)

        alias_list = []
        #also known property
        if "P460" in alias_res["entities"][q]["claims"]:
                claims = alias_res["entities"][q]["claims"]["P460"]
                for link in claims:
                        alias_list.append(link["mainsnak"]["datavalue"]["value"]["id"])
        #different from property
        elif "P1889" in alias_res["entities"][q]["claims"]:
                claims = alias_res["entities"][q]["claims"]["P1889"]
                for link in claims:
                        alias_list.append(link["mainsnak"]["datavalue"]["value"]["id"])
        else:
                return {q:None}

        return {q:alias_list}

    def __normalize_label(self,label):
        '''Function normalizes label:
                * removes everything between "(...)"
                * splits a string into a list of strings if it contains a separator "/"
                * strips a string
        '''
        label_mod = re.sub(r"\(.+?\)","",label)
        label_mod = label_mod.strip()
        if "/" in label_mod:
                label_mod = [x.strip() for x in label_mod.split("/")]

        return label_mod

    def __get_labels(self,q):
        '''Function gets all labels of a given q
        Args:
                q: Q.... to be searched for
                url: url to be used to make a request
        Returns:
                dictionary with q as key and its unique list of labels as value
        '''
        params = urllib.parse.urlencode({
                "action":"wbgetentities",
                "ids":q,
                "props":"labels",
                "utf8":1,
                "format":"json"
        })

        url = f"{self.url}?{params}"
        response = fetch_url(url)
        labels_res = json.load(response)

        labels = []
        for label,label_value in labels_res["entities"][q]["labels"].items():
                value = label_value["value"]
                label = self.__normalize_label(value)
                if isinstance(label,list):
                        for sub_list in label:
                                labels.append(sub_list)
                else:
                        labels.append(label)

        return {q: sorted(set(labels))}


def get_unique_names(original, col):
        '''gets unique values of a given column that are in all sub elements of this column'''
        return_seq = sorted(original[[col]].explode(col).drop_duplicates()[col])
        return return_seq


def parse_names(names):
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
        

if __name__ == "__main__":
        maximumNumberOfThreads = 30
        threadLimiter = BoundedSemaphore(maximumNumberOfThreads)

        original = load_data("data\\source\\united_states_governors.csv")

        unique_names = get_unique_names(original,"governor_split")
        print(len(unique_names))

        names_dict = parse_names(unique_names)
        with open("data\\dict\\names.json", "w",encoding="utf-8") as file:
                file.write(json.dumps(names_dict,indent=2))

        # name = "julia"
        # syn = Synonyms_finder(name)
        # syn.fit()
        # print(syn)
