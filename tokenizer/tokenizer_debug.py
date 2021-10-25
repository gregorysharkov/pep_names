from tokenizer import preprocess_list_into_matrix, load_tokenizer

def main():
    test_strings = ['Adolph Designers of the "Formula1" who participated only in the "Indy-500" Eberhart']
    tokenizer_path = "data\\tokenizer\\20211023_tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    preprocess_list_into_matrix(test_strings,tokenizer,10,10,True)
    pass

if __name__ == '__main__':
    main()