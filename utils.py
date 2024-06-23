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

# Set chat to load
chat_title = 'familychats'
        
def parse_transcript(chat_title):
    """ Parse raw message data into timestamps, senders and messages. Chat title determines what chat to take"""
    
    # Set input file
    input_file = f"./transcripts/transcript_{chat_title}.txt"
    
    # Read text lines
    try:
        with open(input_file, 'r', encoding="utf-8") as txt:
            lines = txt.readlines()
    except:
        raise Exception(f"Unable to read input file at {input_file}")
            
    # Loop through lines to clean
    message_rows = []
    for line in lines[18000:]:
                
        # Find date and time
        pattern = r"\[(\d\d/\d\d/\d\d\d\d),\s(\d\d?:\d\d:\d\d\s?[A-Z]?M?)[^A-Za-z]*\s([A-Za-z]+\s?[A-Za-z]*):\s(.*)"
        match = re.search(pattern, line)
                
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
        
        # Get sentiment from message
        def get_sentiment(text):
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        
        sentiment = get_sentiment(message)
        
        # Create row and add to list of messages
        row = {'date': date, 'time': time_24hr, 'sender': sender, 'message': message, 'gpt_str': gpt_str, 'sentiment': round((sentiment/2 + 0.5),2)}
        message_rows.append(row)    

    # Check the parsing was successful by ensuring message_rows is not empty
    if len(message_rows) == 0:
        raise Exception('Did not find any regex matches in message transcript')
    
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
    df.to_csv(f'processed/parsed_transcript_{chat_title}.csv', index=False, encoding='utf-8-sig')    

def make_conversations(texts, conversation_len = 15, rolling_window = 0.5, max_tokens = 500):
    """Function to create rolling conversations"""
    
    # Preallocate conversations
    conversations = []
    current_tokens = 0
    conversation = []
    
    # Preset indices
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

    # Initialize client to openai
    client = OpenAI(
        api_key = API_KEY
    )

    # Get embeddings
    df['embeddings'] = df['text'].apply(lambda x: client.embeddings.create(input = x, model = "text-embedding-3-large", encoding_format="float").data[0].embedding)

    return df

def preprocessor(chat_title):
    
    # Load csv file
    df = pd.read_csv(f'./processed/parsed_transcript_{chat_title}.csv')
    
    # Tokenizer
    df['n_tokens'] = df['gpt_str'].apply(lambda x: len(tokenizer.encode(x)))
    
    # Get conversations
    conversations = make_conversations(df['gpt_str'], conversation_len=15, rolling_window=0.75)
    
    # Save to a new df with abbreviated texts
    df = pd.DataFrame(conversations, columns = ['text'])
    df['n_tokens'] = df['text'].apply(lambda x: len(tokenizer.encode(x)))
    
    # Find number of tokens to convert
    print(f"Total number of tokens: {df['n_tokens'].sum()}\nRequesting embeddings...")
    
    # Get embeddings
    df = get_embeddings(df)
    df = df.dropna(axis=0)
    
    # Save to CSV - mainly as backup
    try:
        df.to_csv(f'embeddings/embeddings_{chat_title}.csv', index=False)
    except:
        print("Error saving embedding as csv")
        
    # Apply eval to convert embedding to numpy array
    try:
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    except:
        print("Error converting embeddings to np array. Saving with string embeddings...")
        
    # Save as pickle file
    try:
        df.to_pickle(f'embeddings/embeddings_{chat_title}.csv')
    except:
        print("Error saving embedding as pickle")    
    
    
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
        if msg == "":
            continue
        elif msg[:6] != 'Will: ':
            new_array.append('Will: ' + msg)
        else:
            new_array.append(msg)
        
    msg_str = '\n'.join(new_array)
    
    return msg_str

#parse_transcript(chat_title)
#preprocessor(chat_title)

#def checker()