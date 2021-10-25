import urllib
import urllib.request
from time import sleep
from dataclasses import dataclass
from abc import ABC, abstractmethod

def fetch_url(url):
    """
    The function is responsible for fetching a url and getting the response
    if we encouter a URL error or an HTTP error, it waits for some time
    and retries the request until several attempts have failed
    """
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


class abstract_param(ABC):
    base_url:str="https://www.wikidata.org/w/api.php"
    
    def encode(self):
            return f"{self.base_url}?{urllib.parse.urlencode(self._get_param())}"

    @abstractmethod
    def _get_param(self) -> dict:
            pass

@dataclass
class TitleRequestParam(abstract_param):
    """The class that is responsible for generation of an url for wikidata using a title"""
    site:str
    title:str

    def _get_param(self):
        return {
            "action":"wbgetentities",
            "sites":self.site,
            "titles":self.title,
            "format":"json",
            "props":"info|claims|labels",
            "normalize":1,
        }

@dataclass
class IdRequestParam(abstract_param):
    site:str
    id:str

    def _get_param(self):
        return {
            "action":"wbgetentities",
            "sites":self.site,
            "ids":self.id,
            "format":"json",
            "props":"info|claims|labels",
        }


if __name__=="__main__":
    title_param = TitleRequestParam("enwiki","Julia")
    print(title_param.encode())
    id_param = IdRequestParam("enwiki","Q2737173")
    print(id_param.encode())