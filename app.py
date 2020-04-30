#!/usr/bin/python3

import twitter_process
import logging
import json

config = json.load(open('config.json'))

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

twitter_process.start_twitter_stream()
