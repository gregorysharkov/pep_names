import re
import random
import unicodedata as ud
from random import sample, seed
from itertools import product, permutations, chain
from dataclasses import dataclass, field

latin_letters= {}
def is_latin(uchr):
    '''Checks if the character is lating
    '''
    if uchr in latin_letters:
        return latin_letters[uchr]
    else:
        return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))


def only_roman_chars(unistr):
    '''Cheks if the string contains only roman characters
    isalpha suggested by John Machin
    '''
    return all(is_latin(uchr) for uchr in unistr if uchr.isalpha()) 

@dataclass
class TrueSynonymsGenerator():
    """
    The class is responsible for generating true combinations of a given name
    using the provided dictionary
    """
    name: str
    clean_name: str = ""
    name_split: list[str] = field(default_factory=list)
    combinations: list[str] = field(default_factory=list)
    selected_combinations: list[str] = field(default_factory=list)
    n_sample_from_dict: int = None

    def fit(self, dict):
        """
        function generates all possible combinations of the given name
        """
        self._clean_name()
        self._split_name()
        self._generate_combinations(dict)
        pass

    def sample(self,n=None,random_state=1):
        """
        function samples a required number of combinations
        """
        if n:
            seed(random_state)
            self.selected_combinations = sample(self.combinations,min(n,len(self.combinations)))
        else:
            self.selected_combinations = self.combinations
        return self.selected_combinations

    def _clean_name(self):
        """preprocess the name"""
        self.clean_name = re.sub(r"[^\w]"," ",self.name)
        self.clean_name = re.sub("  "," ",self.clean_name)
        self.clean_name = re.sub("  "," ",self.clean_name)
        self.clean_name = self.clean_name.strip()

    def _split_name(self):
        """split name into a list of words"""
        self.name_split = self.clean_name.split(" ")

    def _get_synonyms(self,word:str,dict:dict):
        """get synonyms of a given name, if the dictionary
        does not contain the word, only the word is returned
        """
        return_list = []
        if word in dict:
            return_list = list([word] + dict[word])
       
        return_list = [x for x in return_list if only_roman_chars(x)]
        return_list.sort()
        return_list = list(set([word] + return_list))
        if self.n_sample_from_dict:
            return_list = random.sample(return_list,min(len(return_list),self.n_sample_from_dict),)
        return return_list

    def _generate_combinations(self,dict):
        # get all synomyms
        word_dict = {}
        for word in self.name_split:
            word_dict.update({word:self._get_synonyms(word,dict)})
        
        # get all possible combinations
        synonyms_list = [word_dict[x] for x in self.name_split]
        possible_combinations = list(product(*synonyms_list))
        possible_selection = []
        for el in possible_combinations:
            possible_selection.append(list(" ".join(x) for x in list(permutations(el))))
        
        self.combinations = list(chain(*possible_selection))


if __name__ == '__main__':
    # test_name = "Bill,. Gates Geferson"
    # test_dict = {"Bill": ["William","Guillaume"]}
    import json
    with open("data\\dict\\dictionary_en_fr_ar_ru.yml","r") as f:
        contents = f.read()
        f.close()
    test_dict = json.loads(contents)
    test_name = "Juan Francisco Luis"

    random.seed(20211024)
    test_gen = TrueSynonymsGenerator(test_name,n_sample_from_dict=2)
    test_gen.fit(test_dict)
    test_gen.sample()
    print(len(test_gen.combinations))
    print(len(test_gen.selected_combinations))
    print(repr(test_gen))
    print(test_gen.selected_combinations)