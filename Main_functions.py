import numpy as np
import os
import fnmatch
import re
import cvxpy
from cvxpy.atoms.elementwise.power import power as power
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
from rouge.rouge import rouge_n_sentence_level

def fileProcess(single_data_path, file):
    with open(single_data_path + "/" + file, 'r', encoding="utf8") as infile:
        lines = infile.readlines()
        content = ""
        for line in lines:
            content += line
        infile.close()
    content = content.replace("\n", " ").replace("(", " ").replace(")", " ").replace("  ", " ").replace("  ", " ").replace("\u200c", " ").replace("\ufeff", " ").replace("\u200f", " ")
    sentences = re.split("\.|\?|\!", content)
    for sentence in sentences:
        if len(sentence) < 2:
            sentences.remove(sentence)
    return sentences


def singleSummariesProcess(summary_path,filename):
    for file in os.listdir(summary_path):
        filenames = os.fsdecode(file)
        if fnmatch.fnmatch(filenames, filename + '*'):
            with open(os.path.join(summary_path, filenames), 'r', encoding="utf8") as sumfile:
                lines = sumfile.readlines()
                content = ""
                for line in lines:
                    content += line
                content = content.replace("\n", " ").replace("(", " ").replace(")", " ").replace("  ", " ").replace("  ", " ").replace("\u200c", " ").replace("\ufeff", " ").replace("\u200f", " ").replace("'", "")
                sentences = re.split("\.|\?|\!", content)
                for sentence in sentences:
                    if len(sentence) < 2:
                        sentences.remove(sentence)
                sumfile.close()
    return sentences

#########################################################################
#   This is a function to get the sentences list and return
#       the TFIDF matrix with shape (dimension, num_sentences)
#
#########################################################################
def make_tfidf_matrix(sentences):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    return np.transpose(X).toarray()

#########################################################################
#   In this function the cost function will be minimized according to
#       the W matrix and finally it will be returned.
#   Inputs:
#       1. S: candidate set with size of (d, n)
#       2. L0:
#       3. k: number of sentences that we are interested the
#           results to be
#       4. lam:
#########################################################################
def NAGD(S, epochs, L0, k, lam=0.6):
    n = S.shape[1]
    W_1 = np.random.normal(loc=20, scale=1, size=(n, n))  # weights at the previous time
    W_0 = copy.deepcopy(W_1)               # weights at the current time
    t_1 = 0                 # t - 1 initial value
    t_0 = 1                 # current t initial value
    gamma_i = L0            # the initial value for gamma parameter
    convergence = False     # A constraint that has to be defined

    for i in range(1, epochs):
        alpha = ((t_1) - 1)/(t_0)

        #   This is supposed to be the lookahead W as in Neterow
        X = W_0 + alpha * (W_0 - W_1)
        j = 0
        while True:
            gamma = 2 ** j * gamma_i
            U = X - (1 / gamma_i) * objective_gradient(X, S, lam)
            W_temp = conv_set_projection(U, lam=lam)
            if objective_function(S, W_temp, lam, k) <= objective_function_approximation(W_temp, W_0, X, S, gamma, lam, k):
                gamma_i = gamma
                break
            j += 1

        t_1 = copy.deepcopy(t_0)
        t_0 = cal_t(t_1)
        W_1 = copy.deepcopy(W_0)

        if convergence:
            break
    return W_1

def cal_t(previous_t):
    return 0.5 * (1 + math.sqrt(1 + 4 * previous_t ** 2))

def objective_gradient(X, S, lam):
    X_num = X.shape[0]
    D = np.zeros((X_num, X_num))
    X_norm = np.linalg.norm(X, ord=2, axis=1)
    for i in range(X_num):
        D[i, i] = 0.5 / X_norm[i]
    return 2 * np.matmul(np.transpose(S), np.matmul(S, X)-S) + 2 * lam * np.matmul(D, X)


def objective_function(S, W, lam, k):
    return np.power(np.linalg.norm(np.subtract(S, np.matmul(S, W)), ord="fro"), 2) + lam * norm21(W, k)


def conv_set_projection(U, lam):
    n = U.shape[0]
    W = np.zeros_like(U)
    for i in range(n):
        w_i = cvxpy.Variable(U.shape[1])
        obj = cvxpy.Minimize(0.5 * power(cvxpy.atoms.norm(w_i - U[i, :], 2), p=2) +
                                   lam * cvxpy.atoms.norm(w_i, p=2))
        constr = [w_i >= 0, w_i[i] == 0]
        problem = cvxpy.Problem(obj, constr)
        problem.solve(verbose=False, solver="ECOS")
    return np.array(W)


def conv_set_projection_using_pulp(U):
    pass
    # n = U.shape[0]
    # W = np.zeros_like(U)
    # for i in range(n):
    #     w_i = cvxpy.Variable(U.shape[1])
    #     obj = cvxpy.MiniFalsemize(cvxpy.sum_squares(w_i - U[i, :]) +
    #                          cvxpy.atoms.norm(w_i, p=2))
    #     constr = [w_i >= 0, w_i[i] == 0]
    #     problem = cvxpy.Problem(obj, constr)
    #     problem.solve(verbose=True, solver="ECOS")
    #     print(w_i.value[i] == 0)
    #     W[i, :] = w_i.value
    # return np.array(W)


def objective_function_approximation(W_temp, W_i, X, S, gamma, lam, k):
    asd = objective_function(S, X, lam, k) + \
           np.trace(np.matmul(np.transpose(objective_gradient(X, S, lam)), (W_temp - X))) + \
           0.5 * gamma * np.linalg.norm(W_temp - X, ord="fro")  # changed W_i to W_temp
    return asd


def norm21(W, k):
    norm_2 = np.linalg.norm(W, ord=2, axis=1)
    W[np.where(W <= 0)] = 0
    return np.sum(np.sort(-norm_2)[:k])


#########################################################################
#   In this function the cost function will be minimized according to
#       the W matrix and finally it will be returned.
#   Inputs:
#       1. model: the weights learnt from the Input sentences
#       2. gold_summary: The gold summaries written by agents, used for
#           rouge calculation.
#       3. source_sentences: The source sentences we should summarize
#########################################################################
def Test(model, gold_summary, source_sentences, k, verbose=False):
    ranks = np.linalg.norm(model, ord=2, axis=1)
    calculated_k = 0
    flag = True
    source_sentences = np.asarray(source_sentences)
    gold_sentences_content = " ".join(list(gold_summary))
    len_gold_sentences_content = len(gold_sentences_content.split(" "))
    temp = 0

    print(len_gold_sentences_content)

    while flag:
        temp += len(gold_sentences_content[calculated_k].split())
        if temp >= len_gold_sentences_content:
            flag = False
        calculated_k += 1

    my_summary = source_sentences[np.argsort(-ranks)[:calculated_k]]
    my_summary_content = " ".join(my_summary.tolist())
    print("**********************************")
    print(len(my_summary_content))
    print(len(gold_sentences_content))

    recall1, precision1, rouge1 = rouge_n_sentence_level(my_summary_content, gold_sentences_content, 1)
    recall2, precision2, rouge2 = rouge_n_sentence_level(my_summary_content, gold_sentences_content, 2)

    if verbose:
        print("The calculated results in rouge1 form:"
              "\nrecall:%.2f"
              "\nprecision:%.2f"
              "\nf_measure:%.2f"
              %(recall1, precision1, rouge1))

        print("The calculated results in rouge2 form:"
              "\nrecall2:%.2f"
              "\nprecision2:%.2f"
              "\nf_measure:%.2f"
              %(recall2, precision2, rouge2))

    return [recall1, precision1, rouge1], [recall2, precision2, rouge2]
