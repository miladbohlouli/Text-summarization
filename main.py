from Main_functions import *
import progressbar
import pickle as pk

DATA_PATH = "dataset/Single/Source/DUC"
SUMMARY_PATH = "dataset/Single/Summ/Extractive/"
SAVING_DIR = "cache/"
load = True
saving_dict = dict()
models = []
whole_sentences = []
whole_gold_summaries = []
EPOCHS = 10
L0 = 0.1
LAMBDA = 0.6

if not load:
    for file in progressbar.progressbar(os.listdir(DATA_PATH)[:5]):
        sentences = fileProcess(DATA_PATH, file)
        whole_sentences.append(sentences)
        gold_summaries = singleSummariesProcess(SUMMARY_PATH, file[:-4])
        whole_gold_summaries.append(gold_summaries)
        TFIDF_matrix = make_tfidf_matrix(sentences)
        print(len(gold_summaries))
        models.append(NAGD(TFIDF_matrix, epochs=EPOCHS, L0=L0, lam=LAMBDA, k=len(gold_summaries)))   #   Retrieve the w learnt weights
    saving_dict = {
        "models": models,
        "whole_sentences": whole_sentences,
        "whole_gold_summaries": whole_gold_summaries
    }
    pk.dump(saving_dict, open(SAVING_DIR + "checkpoint.pickle", "wb"))

if load:
    saving_dict = pk.load(open(SAVING_DIR + "checkpoint.pickle", "rb"))
    models = saving_dict["models"]
    whole_sentences = saving_dict["whole_sentences"]
    whole_gold_summaries = saving_dict["whole_gold_summaries"]

    recall_results1 = []
    recall_results2 = []
    precision_results1 = []
    precision_results2 = []
    f_measure_results1 = []
    f_measure_results2 = []
    for i in progressbar.progressbar(range(len(models))):
        rouge1_results, rouge2_results = Test(models[i], whole_gold_summaries[i], whole_sentences[i], len(whole_gold_summaries[i]))
        recall_results1.append(rouge1_results[0])
        recall_results2.append(rouge1_results[0])
        precision_results1.append(rouge1_results[1])
        precision_results2.append(rouge2_results[1])
        f_measure_results1.append(rouge1_results[2])
        f_measure_results2.append(rouge2_results[2])

    print("The average results of the whole documents in rouge1 format:\n"
          "recall:%.3f\n"
          "precision:%.3f\n"
          "f_measure:%.3f\n"
          %(np.average(recall_results1),
            np.average(precision_results1),
            np.average(f_measure_results1)))

    print("The average results of the whole documents in rouge2 format:\n"
          "recall:%.3f\n"
          "precision:%.3f\n"
          "f_measure:%.3f\n"
          %(np.average(recall_results2),
            np.average(precision_results2),
            np.average(f_measure_results2)))




