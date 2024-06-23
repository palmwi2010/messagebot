import re
import pandas as pd
from textblob import TextBlob
import tiktoken
import math
from openai import OpenAI
import time
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables and set api key
load_dotenv()
API_KEY = os.getenv('API_KEY')

# cl100k works with ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")
        
def parse_transcript():
    """ Parse raw message data into timestamps, senders and messages """
    
    # Set input file
    input_file = "transcript.txt"
    
    # Read text lines
    with open(input_file, 'r', encoding="utf-8") as txt:
        lines = txt.readlines()
            
    # Loop through lines to clean
    message_rows = []
    for line in lines[1:]:
                
        # Find date and time
        pattern = r"\[(\d\d/\d\d/\d\d\d\d),\s(\d\d?:\d\d:\d\d\s?[A-Z]?M?)[^A-Za-z]*\s([A-Za-z]+\s?[A-Za-z]*):\s(.*)"
        match = re.search(pattern, line)
                
        # If no match, skip line, otherwise assign to groups
        if match is None:
            continue
        else:
            date =  match.group(1)
            time_12hr = match.group(2)
            sender =  match.group(3)
            message = match.group(4)
            
        # Check it's not a system message
        if sender == "Family chats":
            continue
            
        # Check it's not a system message
        if message.find('\u200e') != -1:
            continue
        
        # Get sentiment from message (for info purposes only)
        def get_sentiment(text):
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        sentiment = get_sentiment(message)
            
        # Format 24hr time
        if len(time_12hr) > 9:
            if time_12hr[-2:] == 'PM' and time_12hr[:2] != '12':
                hrs = str(int(re.match(r'^\d\d?', time_12hr)[0]) + 12)
                time_24hr = re.sub(r'^\d\d?', hrs, time_12hr)[:-3]
            else:
                time_24hr = time_12hr[:-3]
        else:
            time_24hr = time_12hr
            
        # Set string for GPT
        gpt_str = f"{sender}: {message}"
        
        # Create row and add to list of messages
        row = {'date': date, 'time': time_24hr, 'sender': sender, 'message': message, 'gpt_str': gpt_str, 'sentiment': round((sentiment/2 + 0.5),2)}
        message_rows.append(row)    
    
    # Make dataframe
    df = pd.DataFrame(message_rows)
    
    # Time of day: Define the ranges and their corresponding labels
    bins = [0, 6, 11, 14, 18, 24]
    labels = ['Late Night', 'Morning', 'Midday', 'Afternoon', 'Evening']
    
    # Add period to df
    df['hr'] = df['time'].apply(lambda x: int(re.match(r'^\d\d?', x)[0]))
    df['period'] = pd.cut(df['hr'], bins=bins, labels=labels, right=False, include_lowest=True)
    df.drop(['hr'], axis=1, inplace=True)
    
    # Save to csv
    df.to_csv('parsed_transcript.csv', index=False, encoding='utf-8-sig')    

def make_conversations(texts, conversation_len = 15, rolling_window = 0.5, max_tokens = 500):
    """Function to create rolling conversations of window [conversation_len], to train the GPT model"""
    
    # Preallocate conversations
    conversations = []
    current_tokens = 0
    conversation = []
    
    # Preset indices for rolling conversations
    lagger = 0
    leader = 0
    
    # Loop through text and add to conversations
    while lagger < len(texts):
        
        # Set text
        if leader >= len(texts):
            break
        text = texts[leader]
        
        # Check tokens
        n_tokens = len(tokenizer.encode(text))
        
        # If it's not yet at max tokens or set conversation length, add it to the chat
        if n_tokens + current_tokens < max_tokens and len(conversation) < conversation_len:
            
            # Add to the conversation
            conversation.append(text)
            current_tokens += n_tokens
            
            # Add one to leader and lagger
            leader += 1
            lagger += rolling_window
        else:
            # If we're full, reset indices
            if len(conversation) == 0:
                leader += 1
            else:
                leader = math.ceil(lagger)
            lagger = leader
            
            # Add conversation to the list
            conversations.append("\n".join(conversation))
            conversation = []
            current_tokens = 0

    return conversations

def get_embeddings(df):
    """Function to create vector embeddings from a df with text to be embedded in column 'text'"""

    # Initialize client to openai
    client = OpenAI(
        api_key = API_KEY
    )

    # Get embeddings
    df['embeddings'] = df['text'].apply(lambda x: client.embeddings.create(input = x, model = "text-embedding-3-large", encoding_format="float").data[0].embedding)

    return df

def preprocessor():
    
    # Load csv file
    df = pd.read_csv('./parsed_transcript.csv')
    
    # Tokenizer
    df['n_tokens'] = df['gpt_str'].apply(lambda x: len(tokenizer.encode(x)))
    
    # Get conversations
    conversations = make_conversations(df['gpt_str'], conversation_len=15, rolling_window=0.75)
    
    # Save to a new df with abbreviated texts
    df = pd.DataFrame(conversations, columns = ['text'])
    df['n_tokens'] = df['text'].apply(lambda x: len(tokenizer.encode(x)))
    
    # Find number of tokens to convert
    print(f"Total number of tokens: {df['n_tokens'].sum()}\nRequesting embeddings...")
    
    # Get embeddings and drop na
    df = get_embeddings(df)
    df = df.dropna(axis=0)
    
    # Save to csv as a safety measure
    df.to_csv('./embeddings.csv')
    
    # Apply eval and save to pickle file for speed
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    df.to_pickle('...')
    