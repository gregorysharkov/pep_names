import urllib
import urllib.request
from time import sleep

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