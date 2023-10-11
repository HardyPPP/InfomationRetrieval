import json
import math
import os
import re
import sys
import time

from files import porter

# load stopwords
stopwords = set()
with open('./files/stopwords.txt', 'r') as f:
    for line1 in f:
        stopwords.add(line1.rstrip())
p = porter.PorterStemmer()
# load the porter stemmer

k = 1
# BM25 parameter k
b = 0.75
# BM25 parameter b

term_document = {}
# the number of docs that contain a term
# key: term
# value: number of documents that contain the term

term_freq = {}
# the term frequency in each docs
# key: doc_id
# value: dictionary
#     key: term
#     value: term frequency in this doc

doc_len = {}
# key: doc_id
# value: doc_length

idf = {}
# key: term
# value: idf score
# index = {}
# key: doc_id
# value: dict
#   key: term
#   value: bm25 score

BM25_scores = {}
# key: query_id
# value: dict
#   key: doc_id
#   value: similarity between this doc and this query (BM25 score)

all_query = {}
# key: query_id
# value: query_terms[]

output = {}
# dict to be dumped in output.txt
# key: query_id
# value: dict
#   key: doc id
#   value: similarity

all_files = {}
# key: doc_id
# value: doc_path

index = {}


def get_files(file_path):
    all_file = []
    for file in os.listdir(file_path):
        if file == ".DS_Store":
            # find the directory of corpus
            continue
        all_file.append(file)
        # add them into file path collection
        doc_len[file] = 0
    return all_file


def get_large_files(path):
    dirs = os.listdir(path)
    for name in dirs:
        # get sub-dirs.
        if name == ".DS_Store":
            continue
        for file in os.listdir(path + name):
            all_files[file] = path + name + "/" + file
            # build file path and file name, and add it to the list


def create_doc_dictionary():
    """
        calculate the term frequency in each docs
        and calculate the number of docs that contain a term
    """
    print("Reading the corpus...")
    path = "./documents/"
    dirs = os.listdir(path)
    for name in dirs:
        # get sub-dirs.
        if name == ".DS_Store":
            continue
        for file in os.listdir(path + name):
            all_files[file] = path + name + "/" + file
            # open the document
            f = open(path + name + "/" + file, 'r', encoding='UTF-8')
            doc = f.read().lower()
            doc = re.split(r'\W+', doc)
            # remove punctuations
            doc_len[file] = len(doc)
            f.close()
            freq = {}
            for term in doc:
                if term not in stopwords and term != '':
                    # only collect the terms that are not stopword
                    term = p.stem(term)
                    if term not in freq:
                        # find term frequency of this term in this document
                        # if the term is not in the freq, add it to the freq
                        freq[term] = 1
                        if term not in term_document:
                            # find the number of documents that contains this term
                            # if this term is a new one, add it to the dictionary
                            term_document[term] = 1
                        else:
                            # if this term is already in the dictionary, increase its frequency
                            term_document[term] = term_document[term] + 1
                    else:
                        # if the term is in the freq, increase its frequency
                        freq[term] = freq[term] + 1
            term_freq[file] = freq
    print("Complete")


def get_avg_len():
    sum = 0
    count = 0
    for doc_id in doc_len:
        sum = sum + doc_len.get(doc_id)
        count = count + 1
    return sum / count


def create_q_dictionary():
    print("Reading the sample queries...")
    f = open("./files/queries.txt", 'r')
    # open query file
    q = f.read().lower().split("\n")
    for sentence in q:
        # split each query into terms
        if len(sentence) != 0:
            terms = []
            for term in re.split('\W+', sentence)[1:]:
                term = p.stem(term)
                # stem the term
                if term != '':
                    terms.append(term)
                    # add non-empty term into the collection
            all_query[sentence.split()[0]] = terms
    f.close()
    print("Complete")


def generate_BM25_index():
    print("Generating BM25 index...")
    avg_len = get_avg_len()
    BM25_scores = {}
    for doc_id in all_files:
        scores = {}
        score = 0
        for term in term_freq[doc_id]:
            score = ((math.log2(len(doc_len) - term_document[term] + 0.5) - math.log2(term_document[term] + 0.5)) * (
                    term_freq[doc_id][term] * (k + 1) / (
                    term_freq[doc_id][term] + k * ((1 - b) + (b * doc_len[doc_id]) / avg_len))))
            scores[term] = score
        BM25_scores[doc_id] = scores
    data = {
        "BM25_score": BM25_scores,
        "doc_len": doc_len,
        "avg_len": avg_len,
    }
    with open('./index.json', "w") as f:
        # write the data to the file
        json.dump(data, f)
    print("Generation complete")


def load_index():
    if os.path.exists("./index.json"):
        print("Loading index from file...")
        return json.load(open("index.json", "r", encoding='utf8'))
    else:
        return None


def sim(query, index):
    # calculate BM25 scores of each docs for one single query
    scores = {}
    # if we have the index
    for doc_id in index['doc_len']:
        # for each docs
        score = 0
        for term in query:
            # for each term in query
            if term not in index['BM25_score'][doc_id]:
                # only care about the term that exist both in this query and in this doc
                continue
            score = score + index['BM25_score'][doc_id][term]
        scores[doc_id] = score
    return scores


def sim_all(query_dict, index):
    print("Load complete")
    print("Searching...")
    for query_index in query_dict.keys():
        # calculate BM25 scores for al queries
        BM25_scores[query_index] = sim(query_dict.get(query_index), index)
    print("Searching complete")


def sortedDictValues(adict):
    # get the top 10 elements in the sorted dictionary
    items = list(adict.items())
    items.sort(key=lambda x: x[1], reverse=True)
    # sort the dictionary according to its values
    # normalize the BM25 score
    if items[0][1] != 0:
        return [(key, value / (items[0][1])) for key, value in items]
    else:
        return [(key, value / (items[0][1] + 1)) for key, value in items]


def generate_output():
    print("Generating output...")
    result_file = open("./output.txt", 'w', encoding='UTF-8')
    for query_id in BM25_scores:
        dic = sortedDictValues(BM25_scores.get(query_id))
        output[query_id] = dic[:15]
        # get the first 50 results
        i = 0
        while i < len(output[query_id]):
            # if output[query_id][i][1] > 0.7:
            # if the result similarity score is more that 0.7
            result_file.write(query_id + ' ' + output[query_id][i][0] + ' ' + str(i + 1) + ' ' + str(
                format(output[query_id][i][1], '.4f')) + '\n')
            #     write this result into output.txt
            i = i + 1
    print("Generate complete")


def interactive(index_doc):
    query = input('please input a query: ')
    # let user enter a query
    while query != 'QUIT':
        # if it is not QUIT which means user want to stop the system
        q_terms = []
        # store all terms in query
        for term in query.lower().split(' '):
            # split the query that user input
            if term not in stopwords:
                # if the term is not a stopword
                term = p.stem(term)
                # stem it
                q_terms.append(term)
                # add it to the collection

        print('Complete loading')
        result_dic = sortedDictValues(sim(q_terms, index_doc))
        i = 1
        print('Top 15 result for query ' + '[ ' + query + ' ] ' + ' is:')
        for key, value in result_dic[:15]:
            print(str(i) + ' ' + key + ' ' + str(value))
            i = i + 1
        query = input('please input a query: ')


def generate_index():
    time1 = time.process_time()
    # get_large_files("./documents/")
    # time2 = time.process_time()
    # print("get large files")
    # print(time2 - time1)
    # time1 = time.process_time()
    create_doc_dictionary()
    # time2 = time.process_time()
    # print("create doc dic")
    # print(time2 - time1)
    # time1 = time.process_time()
    generate_BM25_index()
    time2 = time.process_time()
    # print("generate index")
    print('it takes', time2 - time1, 'seconds to build index')


def generate_output_with_index():
    create_q_dictionary()
    # index = load_index()
    sim_all(all_query, index)
    generate_output()


if __name__ == '__main__':
    if os.path.exists('./index.json'):
        index = load_index()
    else:
        generate_index()
        index = load_index()
    for arg in sys.argv:
        input_line = sys.argv
        if arg == "-m" and input_line[input_line.index(arg) + 1] == "interactive":
            interactive(index)
        elif arg == "-m" and input_line[input_line.index(arg) + 1] == "automatic":
            generate_output_with_index()
