#!/usr/bin/python3

from pyspark.sql.session import SparkSession
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Tokenizer, CountVectorizer, StopWordsRemover, NGram
from pyspark.mllib.linalg import Vectors as MLLibVectors
import re
import os
import logging
import json
import math
import nltk
from textblob import TextBlob
import requests
import scipy.stats as st

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

config = json.load(open('config.json'))

logging.basicConfig(format='%(asctime)s - [%(filename)s:%(lineno)d] - %(message)s', level=logging.INFO)

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'  # necessary?

model_1 = None
model_2 = None
vader = None


def load_model(sc, model1_path, model2_path):
    global model_1, model_2, vader
    logging.info("Loading Naive Bayes sentiment model 1...")
    model_1 = NaiveBayesModel.load(sc, model1_path)
    logging.info("Naive Bayes sentiment model 1 loaded.")
    logging.info("Loading Naive Bayes sentiment model 2...")
    model_2 = NaiveBayesModel.load(sc, model2_path)
    logging.info("Naive Bayes sentiment model 2 loaded.")
    vader = SentimentIntensityAnalyzer()
    logging.info("VADER analyzer loaded.")


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


def strip_text(text):
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
            'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'can', 'will', 'just', 'don', 'should', 'now']
    text = text.lower()
    text = re.sub("\n", "", text)
    text = re.sub("rt\\s+", "", text)
    text = re.sub("\\s+@\\w+", "", text)
    text = re.sub("@\\w+", "", text)
    text = re.sub("\\s+#\\w+", "", text)
    text = re.sub("#\\w+", "", text)
    text = re.sub("(?:https?|http?)://[\\w/%.-]+", "", text)
    text = re.sub("(?:https?|http?)://[\\w/%.-]+\\s+", "", text)
    text = re.sub("(?:https?|http?)//[\\w/%.-]+\\s+", "", text)
    text = re.sub("(?:https?|http?)//[\\w/%.-]+", "", text)
    text = re.split("\\W+", text)
    for i in range(0, len(stop)):
        if stop[i] in text:
            text = remove_values_from_list(text, stop[i])
    return text


def normalize(score, alpha=0.15):  # borrowed from VADER sentiment analysis algorithm
    norm_score = score / math.sqrt((score * score) + alpha)
    return norm_score


def average_values(values):
    return sum(values) / len(values)


def get_140_sentiment(statuses):
    data_text = []
    for status_text in statuses:
        data_text.append({'text': status_text})
    data = {'data': data_text}
    res = requests.post(url='http://www.sentiment140.com/api/bulkClassifyJson', data=data)
    scores = []
    if res:
        for entry in res.json()['data']:
            score = (entry['polarity'] / 4)  # shift 0-4 to 0-1
            scores.append(score)
    return scores


def get_textblob_sentiment(statuses):
    scores = []
    for status_text in statuses:
        analysis = TextBlob(status_text)
        # return str(analysis.sentiment.polarity)
        score = (analysis.sentiment.polarity + 1) / 2  # -1 to 1 -> 0-1
        scores.append(score)
    return scores


def get_vader_sentiment(statuses):
    scores = []
    for status_text in statuses:
        score = vader.polarity_scores(status_text)['compound']  # between -1 and 1
        score = (score + 1) / 2  # shift to 0-1 scale
        scores.append(score)
    return scores


def combine_weighted_scores_after(scores=[], weights=[]):
    final = 0.0
    for i in range(0, len(scores)):
        final += (scores[i] * weights[i])
    return final


def combine_weight_scores_before(scores, weights):
    combined_scores = []
    for i in range(0, len(scores[0])):  # for each entry in nb_scores
        score = 0.0
        for j in range(0, len(scores)):  # for all score types
            score += (scores[j][i] * weights[
                j])  # ex. e = [[n1,n2],[v1,v2],[t1,t2]], w = [nw,vw,tw] -> e[2][1] * w[2] = t1 * tw
        combined_scores.append(score)
    return combined_scores


def get_nb_sentiment_model2(statuses):
    scores = []
    for status_text in statuses:
        clean_text = strip_text(status_text)
        if clean_text:
            score = (model_2.predict(HashingTF().transform(clean_text)) / 4)  # shift 0-4 to 0-1
            scores.append(score)
    return scores


def get_nb_sentiment_model1(raw_statuses, sc):
    statuses = []
    spark = SparkSession(sc)
    for status in raw_statuses:
        statuses.append(strip_text(status))
    if statuses:
        logging.info("Analyzing " + str(len(statuses)) + " Twitter statuses...")
        template = []
        for i in range(0, len(statuses)):
            template.append(i)
        tweets = sc.parallelize(template)
        dataset = tweets.map(lambda r: (0, statuses[r]))
        status_df = spark.createDataFrame(dataset, ["sentiment", "tweet"])

        """code block below heavily adapted from https://github.com/sohaibomr/tweet-sentiment-pyspark"""
        remover = StopWordsRemover(inputCol="tweet", outputCol="filtered")
        filtered_df = remover.transform(status_df)
        # now make 2-gram model
        ngram = NGram(n=2, inputCol="filtered", outputCol="2gram")
        gram_df = ngram.transform(filtered_df)
        # now make term frequency vectors out of data frame to feed machine
        hashingtf = HashingTF(inputCol="2gram", outputCol="tf", numFeatures=20000)
        tf_df = hashingtf.transform(gram_df)
        # convert dataframe to rdd, to make a LabeledPoint tuple(label, feature, vector) for machine
        tf_rdd = tf_df.rdd
        test_dataset = tf_rdd.map(lambda x: LabeledPoint(float(x.sentiment), MLLibVectors.fromML(x.tf)))
        predictions = test_dataset.map(lambda x: model_1.predict(x.features))
        return predictions.collect()
    return None


def normalize_to_bell_curve(score):
    sd_wt = 0.16666
    prob = st.norm.cdf((score - 0.5) / sd_wt)
    return prob


def get_sentiments(statuses, sc, nb_weight=0.34, vader_weight=0.33, textblob_weight=0.33, combine_after_average=False,
                   normalize=False):
    final_score = 0.0
    vader_scores = get_vader_sentiment(statuses)  # 0-1
    nb_scores = get_nb_sentiment_model1(statuses, sc)  # 0-1
    # nb_scores_2 = get_nb_sentiment_model2(statuses)  # 0-1  MODEL WON'T OPERATE PROPERLY
    textblob_scores = get_textblob_sentiment(statuses)  # 0-1
    # sent140_scores = get_140_sentiment(statuses)  # 0-1  API NOT RETURNING JSON PROPERLY
    # exit()
    log_to_sql(statuses=statuses, nb_scores=nb_scores, vader_scores=vader_scores, textblob_scores=textblob_scores,
               sc=sc)
    if combine_after_average:  # average individual scores types, then apply weights
        combined_scores = []
        nb_score = average_values(nb_scores)
        if nb_score:
            combined_scores.append(nb_score)
        vader_score = average_values(vader_scores)
        if vader_score:
            combined_scores.append(vader_score)
        textblob_score = average_values(textblob_scores)
        if textblob_score:
            combined_scores.append(textblob_score)
        # sent140_score = average_values(sent140_scores)
        # if sent140_score:
        #    combined_scores.append(sent140_score)
        # nb_score_2 = average_values(nb_scores_2)
        # if nb_score_2:
        #    combined_scores.append(nb_score_2)
        final_score = combine_weighted_scores_after(scores=combined_scores,
                                                    weights=[nb_weight, vader_weight, textblob_weight])
    else:  # apply weights, then average final scores
        final_scores = combine_weight_scores_before(scores=[nb_scores, vader_scores, textblob_scores],
                                                    weights=[nb_weight, vader_weight, textblob_weight])
        final_score = average_values(final_scores)
    if normalize:
        return normalize_to_bell_curve(final_score)  # this will stretch the data to match a bell curve, exaggerating scores that are otherwise stuck around 0.5
    return final_score


def log_to_sql(statuses, nb_scores, vader_scores, textblob_scores, sc):
    spark = SparkSession(sc)
    template = []
    for i in range(0, len(statuses)):
        template.append(i)
    scores = sc.parallelize(template)
    dataset = scores.map(lambda j: (statuses[j], nb_scores[j], vader_scores[j], textblob_scores[j]))
    score_df = spark.createDataFrame(dataset, ["text", "nb_score", "vader_score", "textblob_score"])
    score_df.registerTempTable('score_breakdown')


def get_score_breakdown(sqlContext):
    history = sqlContext.sql('Select * from score_breakdown')
    history_df = history.toPandas()
    return history_df
