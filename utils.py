import re
import pandas as pd
from textblob import TextBlob
import tiktoken
import math
from openai import OpenAI
import time
import numpy as np
import os
from dotenv import load_dotenv # type: ignore

# Load environment variables and set api key
load_dotenv()
self_name = os.getenv('RESPONDER')

    
def timer(start, func = "Ran", run_timer = True):
    if run_timer:
        print(f"{func} in {round(time.time() - start,2)} seconds")


def delay(x):
    time.sleep(x)
    
    
def clean_output(msg):
    
    # Split into an array of msgs
    msg_array = msg.split('\n')
    
    # Remove any mention of Will:
    new_array = []
    for msg in msg_array:
        if msg == "" or msg == ".":
            continue
        
        # Remove full stops from messages
        if msg[-1] == '.':
            msg = msg[:-1]
            
        # Add sender name to the message if not already there
        match = re.search(r'([A-Za-z]+\s?[A-Za-z]*):.*', msg)
        if match is None:
            new_array.append(f'{self_name}: ' + msg)
        else:
            new_array.append(msg)
        
    msg_str = '\n'.join(new_array)
    
    return msg_str