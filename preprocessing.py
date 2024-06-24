import re
import pandas as pd
from textblob import TextBlob
import tiktoken
import math
from openai import OpenAI
import numpy as np
import os
import sys
from dotenv import load_dotenv

# Load environment variables, set api key and initialize tokenizer
load_dotenv()
API_KEY = os.getenv('API_KEY')
tokenizer = tiktoken.get_encoding("cl100k_base")

def main():
    """Function to control preprocessing script. Set with command line whether to parse (parse only), embed (embed only)
      preprocess (do everything), init (initalize dirs) or check(list chat titles). Chat title to preprocess also set by command line. See help for more details"""

    # Control for base command line inputs
    control_cli()

    # Set chat title and max conversations
    chat_title = sys.argv[2]

    if sys.argv[1] == 'parse' or sys.argv[1] == 'preprocess':
        # Parse the transcript
        parse_transcript(chat_title)

    if sys.argv[1] == 'embed' or sys.argv[1] == 'preprocess':
        # Parse the transcript
        preprocessor(chat_title)#, max_conversations = 100)


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
    for line in lines:
                
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

        # Ignore if it contains a web url
        match_web = re.search(r"(http).*", line)
        if match_web is not None:
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

def make_conversations(texts, conversation_len = 15, rolling_window = 0.5, max_tokens = 500, max_conversations = 1e6):
    """Function to create rolling conversations"""
    
    # Preallocate conversations
    conversations = []
    current_tokens = 0
    conversation = []
    
    # Preset indices
    lagger = 0
    leader = 0
    
    # Loop through text and add to conversations
    while lagger < len(texts) and len(conversations) < max_conversations:
        
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
    """Function to get embeddings from a df with text to be embedded in column text"""

    # Initialize client to openai
    client = OpenAI(
        api_key = API_KEY
    )

    # Get embeddings
    df['embeddings'] = df['text'].apply(lambda x: client.embeddings.create(input = x, model = "text-embedding-3-large", encoding_format="float").data[0].embedding)

    return df

def preprocessor(chat_title):
    """Function to run full preprocessor from a csv transcript to an embedded df"""

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
        df = pd.read_csv(f'embeddings/embeddings_{chat_title}.csv')
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    except:
        print("Error converting embeddings to np array. Saving with string embeddings...")
        
    # Save as pickle file
    try:
        df.to_pickle(f'embeddings/embeddings_{chat_title}.pkl')
    except:
        print("Error saving embedding as pickle")


def check_chat_titles():
    """Function to check for all possible chat titles"""

    # Prellocate titles
    titles = []

    # Get filenames
    for filename in os.listdir('transcripts/'):
        
        # Find title and add to titles
        match = re.search(r'_([A-Za-z]+).txt', filename)
        if match is not None:
            titles.append(match.group(1))
        else:
            print(f"Could not find chat title for transcript {filename}")
        
    return titles


def init_dirs(verbose = True):
    """Function to initialize directories for processed, transcripts and embeddings (useful following git clone)"""

    # List of directories to create
    directories = ["transcripts", "processed", "embeddings"]

    # Loop through the list and create each directory if it doesn't exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            if verbose:
                print(f"Directory '{directory}' created.")
        else:
            if verbose:
                print(f"Directory '{directory}' already exists.")


def check_dirs():
    """Function to check if correct directory structure is in place with at least one valid transcript"""

    # List of directories to create
    directories = ["transcripts", "processed", "embeddings"]

    # Loop through the list and create each directory if it doesn't exist
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Correct directory structure not in place. Folder must contain directories {'. '.join(directories)}")
            print("Run init operation to add required directories, and ensure transcripts contains a transcript of type [...[chat_title].txt]")
            return False
        
    if len(check_chat_titles()) == 0:
        print("No valid transcripts found in transcripts folder. Please upload a valid transcript with file format [...[chat_title].txt], e.g. transcript_will.txt")
        return False

    return True


def control_cli():
    """Function to control for command line inputs"""

    if len(sys.argv) < 2:
        print("Usage: python script.py <operation> <chat_title> where operation is parse, embed, preprocess, init, check or help, and chat_title is chat to preprocess")
        sys.exit(1)
    elif sys.argv[1] not in ["parse", "embed", "preprocess", "init", "check", "help"]:
        print("Usage: python script.py <operation> <chat_title> where operation is parse, embed, preprocess, init, check or help, and chat_title is chat to preprocess")
        sys.exit(1)

    # If argv1 is init, just initialize directories and return
    if sys.argv[1] == "init":
        init_dirs()
        sys.exit(1)

    # If argv1 is help, print help message
    if sys.argv[1] == "help":
        print("""Usage: python script.py <operation> <chat_title?>\n\nOperations are one of:\nhelp: Get help on script usage\ninit: Initialize file storage directories
check: See current available chats to embed\nparse: Parse chat to csv\nembed: Get embeddings from chat csv\npreprocess: Parse and then embed\n
For operations parse, embed and preprocess, a <chat_title> parameter is required which is the name of the chat to embed - use check to see options""")
        sys.exit(1)


    # Otherwise, check that the correct folder structure is in place
    if not check_dirs():
        sys.exit(1)

    
    # If check, display the transcript chat title options
    if sys.argv[1] == "check":
        titles = '\n'.join(check_chat_titles())
        print(f"Chat options found:\n{titles}")
        sys.exit(1)

    # Otherwise, there should be an additional input for the chat title within the command line inputs
    if len(sys.argv) < 3:
        print(f"For operations parse, embed or preprocess, enter a chat title. Current chat titles found:\n{', '.join(check_chat_titles())}")
        sys.exit(1)
    elif sys.argv[2] not in check_chat_titles():
        print(f"Chat title not found. Current chat titles found:\n{', '.join(check_chat_titles())}")
        sys.exit(1)

main()