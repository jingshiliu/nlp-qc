
punctuations = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

punctuations_set = set(punctuations)

with open('./data/imdb.vocab') as vocab_file:

    appeared_punc = {}
    for line in vocab_file:
        line = line.splitlines()[0]
        print(line)
        for c in line:
            if c in punctuations_set:
                appeared_punc[c] = line
    print(appeared_punc)