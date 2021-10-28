import random

def abbreviate_random_word(string:str) -> str:
    """function transforms one word into an abbreviation. Example Grigory -> G."""
    if not string:
        return None
    
    if isinstance(string,float):
        return None

    string_split = string.split(" ")
    if len(string_split)<2:
        return string

    word_index = random.randint(0, len(string_split)-1)
    if len(string_split[word_index]) > 1:
        string_split[word_index] = string_split[word_index][0]+"."

    return " ".join(string_split)

def remove_random_word(string:str) -> str:
    '''function removes a random word from the string'''
    if not string:
        return None

    if isinstance(string,float):
        return None


    string_split = string.split(" ")
    if len(string_split) < 3:
        return string
    
    word_index = [random.randint(0, len(string_split))]
    string_split = [x for i,x in enumerate(string_split) if i not in word_index]

    return ' '.join(string_split)


def main():
    random.seed(20211028)

    test_strings = ["Sharkov Grigory Andreevich",None,"Sharkov Grigory"]
    abbreviations = list(map(abbreviate_random_word,test_strings))
    shorts = list(map(remove_random_word,test_strings))
    print(f"{abbreviations=}\n{shorts=}")
    pass

if __name__ == '__main__':
    main()
