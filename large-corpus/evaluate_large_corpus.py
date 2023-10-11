import json
import math
import os
import re
import sys
import time

from files import porter



relevant_docs = {}
# the relevance score of relevant docs
# key: query_id
# value: dict
#   key: doc_id
#   value: relevance

relevant_docID_list = {}
# doc ids of relevant docs
# key: query_id
# value: list of relevant doc id

judged_docID_list = {}  # doc id that is judged

result = {}
# doc id in output.txt and its score
# key: query_id
# value: dict
#   key: doc_id
#   value: relevance

result_docID_list = {}
# doc ids in result
# key: query_id
# value: doc_ids[]



def read_rels():
    # read the qrels.txt and create the dictionaries for relevant docs for each queries
    f = open('./files/qrels.txt')
    line1 = f.readline().strip()
    doc_relevance = {}
    # key: doc id
    # value: relevance score
    relevant_id_list = []
    judged_id_list2 = []
    while line1:
        info = line1.split(" ")
        relevance = info[3]
        doc_id = info[2]
        q_id = info[0]
        if relevance != "0":
            # if the relevance score is not 0, this doc is relevant, put it into the list
            relevant_id_list.append(doc_id)
            doc_relevance[doc_id] = relevance
        # no matter what relevance score a doc has, it is judged if the doc is in the qrels.txt
        judged_id_list2.append(doc_id)
        line1 = f.readline().strip()
        # move to the next line
        if line1.split(" ")[0] != q_id:
            # if the query id changed,
            # it means that we have finished building the list of relevant docs for the previous query
            relevant_docs[q_id] = doc_relevance
            relevant_docID_list[q_id] = relevant_id_list
            judged_docID_list[q_id] = judged_id_list2
            # record the relevant or judged docs for the previous query id
            doc_relevance = {}
            relevant_id_list = []
            judged_id_list2 = []
            # clear the dic and id_list for the next query
    f.close()


def read_output():
    # read the output.txt and create dictionary for retrieve result
    f2 = open('./output.txt')
    line2 = f2.readline().strip()
    # open the result file
    doc_relevance = {}
    # store all the results and their similarity scores for one query
    # key: doc id
    # value: relevance score
    id_list = []
    # doc ids
    while line2:
        # read the result
        q = line2.split(" ")[0]
        # query id
        doc_relevance[line2.split(" ")[1]] = line2.split(" ")[3]
        # the similarity score of doc
        id_list.append(line2.split(" ")[1])
        # doc ids in result
        line2 = f2.readline().strip()
        # read the next line
        if line2.split(" ")[0] != q:
            # if the query id changed in next line,
            # it means that we have finished building the list of result for the previous query
            result[q] = doc_relevance
            # put the result dict of this query in result
            result_docID_list[q] = id_list
            # put the doc ids in result of this query in result_docID_list
            doc_relevance = {}
            id_list = []
            # clear all the collection
    f2.close()


def precision(rel, res):
    # calculate the ratio of relevant docs in result docs
    relevant_count = 0
    # the number of relevant docs in result
    for i in res:
        if i in set(rel):
            # if a doc is in rel, it means it is relevant
            relevant_count = relevant_count + 1
    return relevant_count / len(res)


def recall(rel, res):
    # calculate the ratio of retrieved relevant docs
    relevant_count = 0
    # the number of relevant docs in result
    for i in res:
        if i in set(rel):
            # if a doc is in rel, it means it is relevant
            relevant_count = relevant_count + 1
    return relevant_count / len(rel)


def r_precision(rel, res):
    relevant_count = 0
    # the number of relevant docs in result
    for i in res[:len(rel)]:
        # if the number of relevant docs is greater than the number of docs in result
        # then iterate all the result docs
        # if the number of relevant docs is less than the number of docs in result
        # then iterate the first n docs (n is the number of relevant docs)
        if i in set(rel):
            # if a doc is in rel, it means it is relevant
            relevant_count = relevant_count + 1
    return relevant_count / len(rel)


def precisionAt10(rel, res):
    relevant_count = 0
    # the number of relevant docs in result
    i = 0
    while i < 10:
        if i < len(res):
            if res[i] in set(rel):
                # if a doc is in rel, it means it is relevant
                relevant_count = relevant_count + 1
        i = i + 1
    return relevant_count / min(10, len(res))


def map(rel, res):
    relevant_score = 0
    # final score
    relevant_count = 0
    # count of relevant documents
    i = 0
    while i < len(res):
        if res[i] in set(rel):
            # if a doc is in rel, it means it is relevant
            relevant_count = relevant_count + 1
            relevant_score = relevant_score + (relevant_count / (i + 1))
            # average precision formula
        i = i + 1
    # return the mean average precision
    return relevant_score / len(rel)


def bpref(rel, res, judge):
    unrelevant_count = 0
    # count of non-relevant documents
    score = 0
    # evaluation score
    i = 0
    while i < len(res):
        if (res[i] not in set(rel)) and (res[i] in set(judge)):
            # if a doc is not in rel and is in judge, it means this doc is non-relevant
            if unrelevant_count < len(rel):
                unrelevant_count = unrelevant_count + 1
        elif res[i] in set(rel):
            # if a doc is in rel, it means it is relevant
            score = score + (1 - (unrelevant_count / len(rel)))
            # bpref formula
        i = i + 1
    return score / len(rel)


def IDCG(rel, res, rel_score):
    # for idcg, we assume that all docs having highest relevance score are ranked at top
    # so in this example, all dos got 4 scores are ranked before the ones got 3,
    # all docs got 3 are ranked before the ones got 2
    # ......
    i1 = 0
    # the number of docs that got 1 score
    i2 = 0
    # the number of docs that got 2 score
    i3 = 0
    # the number of docs that got 3 score
    i4 = 0
    # the number of docs that got 4 score
    idcg = 0
    idcgs = []
    dcg = 0
    for i in rel:
        # calculate the number of docs for each scores
        if rel_score.get(i) == '3':
            i3 += 1
        elif rel_score.get(i) == '2':
            i2 += 1
        elif rel_score.get(i) == '1':
            i1 += 1
        elif rel_score.get(i) == '4':
            i4 += 1
    k = 0
    while k < i4:
        # doc got 4 ranked at top
        if k + 1 == 1:
            # if it is the first doc
            idcg = idcg + 4
        else:
            # if it is not the first doc
            idcg = idcg + (4 / math.log2(k + 1))
        idcgs.append(idcg)
        k = k + 1
    while k < i4 + i3:
        #  docs got 3 are ranked lower than docs got 4 but higher that the ones got 2
        if k + 1 == 1:
            # if it is the first doc
            idcg = idcg + 3
        else:
            # if it is not the first doc
            idcg = idcg + (3 / math.log2(k + 1))
        idcgs.append(idcg)
        k = k + 1
    while k < i4 + i3 + i2:
        #  docs got 2 are ranked lower than docs got 3 but higher that the ones got 1
        if k + 1 == 1:
            # if it is the first doc
            idcg = idcg + 2
        else:
            # if it is not the first doc
            idcg = idcg + (2 / math.log2(k + 1))
        idcgs.append(idcg)
        k = k + 1
    while k < i4 + i3 + i2 + i1:
        #  docs got 1 are ranked at the bottom
        if k + 1 == 1:
            # if it is the first doc
            idcg = idcg + 1
        else:
            # if it is not the first doc
            idcg = idcg + (1 / math.log2(k + 1))
        idcgs.append(idcg)
        k = k + 1
    return idcgs


def NDCG_AT10(rel, res, rel_score):
    cg = 0
    dcg = 0

    i = 0
    k = 0
    while i < 10:
        if i < len(res):
            if res[i] in set(rel):
                if i != 0:
                    dcg = dcg + (int(rel_score.get(res[i])) / math.log2(i + 1))
                else:
                    dcg = dcg + (int(rel_score.get(res[i])) / 1)
        i = i + 1
    if 10 > len(IDCG(rel, res, rel_score)):
        return dcg / IDCG(rel, res, rel_score)[len(IDCG(rel, res, rel_score)) - 1]
    else:
        return dcg / IDCG(rel, res, rel_score)[i - 1]


def evaluate():
    # evaluate the result

    precision_score = 0
    recall_score = 0
    r_precision_score = 0
    map_score = 0
    p_at_10_score = 0
    bpref_score = 0
    ndcg_at_10_score = 0

    for query_id in result.keys():
        # # calculate metrics
        precision_score += precision(relevant_docID_list[query_id], result_docID_list[query_id])
        recall_score += recall(relevant_docID_list[query_id], result_docID_list[query_id])
        r_precision_score += r_precision(relevant_docID_list[query_id], result_docID_list[query_id])
        map_score += map(relevant_docID_list[query_id], result_docID_list[query_id])
        p_at_10_score += precisionAt10(relevant_docID_list[query_id], result_docID_list[query_id])
        bpref_score += bpref(relevant_docID_list[query_id], result_docID_list[query_id], judged_docID_list[query_id])
        ndcg_at_10_score += NDCG_AT10(relevant_docID_list[query_id], result_docID_list[query_id],
                                      relevant_docs[query_id])

    precision_score /= len(result)
    recall_score /= len(result)
    r_precision_score /= len(result)
    map_score /= len(result)
    p_at_10_score /= len(result)
    bpref_score /= len(result)
    ndcg_at_10_score /= len(result)
    # calculate average score

    print(f'Precision    {precision_score}')
    print(f'Recall       {recall_score}')
    print(f'Precision@10 {p_at_10_score}')
    print(f'R-Precision  {r_precision_score}')
    print(f'MAP          {map_score}')
    print(f'b_pref       {bpref_score}')
    print(f'NDCG_score   {ndcg_at_10_score}')
    # print the result



def make_evaluation():
    if os.path.exists("output.txt"):
        print("Evaluation results:")
        read_rels()
        read_output()
        evaluate()
    else:
        print("output.txt is not found, please run the search program first")


if __name__ == '__main__':
    make_evaluation()

