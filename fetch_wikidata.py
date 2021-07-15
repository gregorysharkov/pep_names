import re
from typing import Match
import urllib
import json
import requests
from threading import *
from time import *
import time
import pandas as pd


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
        self.instances = ["Q101352","Q202444","Q3409032","Q11879590","Q12308941","Q1243157","Q1076664","Q110874"]
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
        '''
        credentials = open(".\conf\local\credentials.txt","r")
        self.key = credentials.readline()[:-1]
        self.user = credentials.readline()[:-1]
        self.pwd = credentials.readline()[:-1]
        credentials.close()

    def __install_openner(self):
        '''Create a proxy gate for requests
        '''
        proxy_url = "@proxy-sgt.si.socgen:8080"
        proxy = {"https": "https://" + self.user + ":" + self.pwd + proxy_url}
        proxy_support = urllib.request.ProxyHandler(proxy)
        opener = urllib.request.build_opener(proxy_support)
        urllib.request.install_opener(opener)

    def __get_qs_norm(self,text):
        '''Function fetches all qs associated with the given string in the given sites
        Args:
                string: string to be searched for
                url: url used inthe search query
                sites: wiki sites to be used
        returns:
                list of Q....
        '''
        for site in self.sites:
                params = urllib.parse.urlencode({
                                "action" : "wbgetentities",
                                "sites" : site,
                                "titles": text,
                                "props" : "descriptions|claims",
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
        if "P460" in alias_res["entities"][q]["claims"]:
                claims = alias_res["entities"][q]["claims"]["P460"]
                for link in claims:
                        alias_list.append(link["mainsnak"]["datavalue"]["value"]["id"])

                return {q:alias_list}
        else:
                return {q:None}

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

        return {q: list(set(labels))}

    def __get_alias(self,q):
        '''Function gets alias names for qs
        Args:
                qs: list of qs to be searched
                url: url to be used in the search

        Returns:
                list of names located in P460
        '''
        params = urllib.parse.urlencode({
                "action":"wbgetentities",
                "ids":q,#"|".join([q,"P460"]),
                "props":"info|claims",
                "format":"json"
        })

        url = f"{self.url}?{params}"
        response = fetch_url(url)
        alias_res = json.load(response)

        alias_list = []
        if "P460" in alias_res["entities"][q]["claims"]:
                claims = alias_res["entities"][q]["claims"]["P460"]
                for link in claims:
                        alias_list.append(link["mainsnak"]["datavalue"]["value"]["id"])

                return {q:alias_list}
        else:
                return {q:None}


def load_data(path,debug=True):
        '''Function loads data from the given path
        '''
        original = pd.read_csv(path,sep=";")
        original[["surname","name"]] = original.bg_name.str.split(', ',expand=True)
        original["name_split"] = original.name.str.split(" ")
        original["surname_split"] = original.surname.str.split(" ")
        if debug:
                print(original.shape)
                print(original.head())
        return original


def get_unique_names(original, col):
        '''gets unique values of a given column that are in all sub elements of this column'''
        return_seq = list(original[[col+"_split"]].explode(col+"_split").drop_duplicates()[col+"_split"])
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
        # maximumNumberOfThreads = 30
        # threadLimiter = BoundedSemaphore(maximumNumberOfThreads)

        # original = load_data(".\\notebooks\\wikidata\\data\\bg_names.csv")
        # unique_names = get_unique_names(original,"name")
        # unique_surnames = get_unique_names(original,"surname")
        # print(len(unique_names),len(unique_surnames))

        # names_dict = parse_names(unique_names)
        # with open(".\\notebooks\\wikidata\\data\\dict\\names.json", "w",encoding="utf-8") as file:
        #         file.write(json.dumps(names_dict,indent=2))

        # surnames_dict = parse_names(unique_surnames)
        # with open(".\\notebooks\\wikidata\\data\\dict\\surnames.json", "w",encoding="utf-8") as file:
        #         file.write(json.dumps(surnames_dict,indent=2))

        name = "Sharkov"
        syn = Synonyms_finder(name)
        syn.fit()
        print(syn)