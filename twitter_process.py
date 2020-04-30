#!/usr/bin/python3

from tweepy.api import API
from tweepy.auth import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import Stream
import logging
import json
import time
import sentiment
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark import SparkContext
import os
import painting
import sys
import datetime
import boto3

config = json.load(open('config.json'))

logging.basicConfig(format='%(asctime)s - [%(filename)s:%(lineno)d] - %(message)s', level=logging.INFO)

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'  # necessary?

# Twitter API Credentials
consumer_key = config['twitter']['consumer_key']
consumer_secret = config['twitter']['consumer_secret']
access_token = config['twitter']['access_token']
access_secret = config['twitter']['access_secret']

sc = SparkContext(appName="TwitterMakesArt")
spark = SparkSession(sc)
sqlContext = SQLContext(sc)

collected_tweets = []


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def filter_tweet(status):
    if True:
        return status.text
    else:
        return None


class TweetsListener(StreamListener):

    def __init__(self, api=None):
        self.api = api or API()

    def on_status(self, status):
        global collected_tweets
        status_text = filter_tweet(status)
        if status_text:
            collected_tweets.append(status_text)
            # print(status_text)

    def on_error(self, status_code):
        logging.error(f"{status_code} status code from Twitter.")
        if status_code == 420:
            logging.error("Rate-limited, halting connection...")
            return False
        logging.info("Continuing connection, but backing off...")
        return True


def color_picture(sentiment_score, image):
    rgb = painting.get_rgb_from_bw_percentage(sentiment_score, ratios={'R': config['image']['R_ratio'],
                                                                       'G': config['image']['G_ratio'],
                                                                       'B': config['image']['B_ratio']})
    new_image = image._check_if_need_reset()
    if new_image:
        # upload_to_s3(image.save_path)
        image = new_image
        logging.info(f"Created new image: {image.save_path}")
    logging.info(f"Coloring [{image.current_x},{image.current_y}] on {image.save_path}: color {rgb}")
    image.color_next_pixel(rgb=rgb)
    return image


def analyze_tweets():
    if collected_tweets:
        if len(collected_tweets) > 100:  # only analyze first 100 tweets
            sentiments_score = sentiment.get_sentiments(statuses=collected_tweets[:100], sc=sc,
                                                        nb_weight=config['sentiment']['nb_weight'],
                                                        vader_weight=config['sentiment']['vader_weight'],
                                                        textblob_weight=config['sentiment']['textblob_weight'],
                                                        combine_after_average=True, normalize=True)
        else:
            sentiments_score = sentiment.get_sentiments(statuses=collected_tweets, sc=sc,
                                                        nb_weight=config['sentiment']['nb_weight'],
                                                        vader_weight=config['sentiment']['vader_weight'],
                                                        textblob_weight=config['sentiment']['textblob_weight'],
                                                        combine_after_average=True, normalize=True)
        logging.info(f"Combined sentiment: {sentiments_score}")
        return sentiments_score
    else:
        logging.error("No tweets collected. Halting program...")
        sys.exit(1)


def upload_to_s3(file_path):
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=config['aws']['key_id'],
            aws_secret_access_key=config['aws']['key_secret'],
            aws_session_token=config['aws']['session_token']
        )
        s3.upload_file(file_path, config['aws']['s3_bucket'], f"{get_timestamp()}.png")
    except Exception as e:
        logging.error(e)


def start_twitter_stream():
    global collected_tweets
    s3 = boto3.resource('s3')
    sentiment.load_model(sc, config['sentiment']['model1_path'], config['sentiment']['model2_path'])
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    twitter_stream = Stream(auth, TweetsListener())
    twitter_stream.filter(track=config['twitter']['keywords_to_track'], is_async=True)
    logging.info("Twitter stream connected, collecting tweets...")
    image = painting.TwitterImage(x_dim=config['image']['X_dim'], y_dim=config['image']['Y_dim'],
                                  filename=config['image']['save_path'])
    while True:
        time.sleep(config['twitter']['stream_interval'])
        sentiments_score = analyze_tweets()
        print(sentiment.get_score_breakdown(sqlContext=sqlContext))
        image = color_picture(sentiment_score=sentiments_score, image=image)
        collected_tweets.clear()
