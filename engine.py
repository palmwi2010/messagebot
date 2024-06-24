"""Script to turn embeddings df into a response"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from openai import OpenAI
import time
from utils import timer, clean_output
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Config - whether to run timers
run_timers = False

# Initialize OpenAI client
#start = time.time()
API_KEY = os.getenv('API_KEY')
client = OpenAI(
    api_key = API_KEY
)
#timer(start, func = "Openened OpenAI connection", run_timer = run_timers)


def create_context(question, df, max_len = 1000):
    """Create a context for a question by finding the most similar context from the dataframe"""

    start = time.time()
    # Get embeddings from the question
    q_embeddings = client.embeddings.create(input = question, model="text-embedding-3-large").data[0].embedding
    timer(start, func='Found prompt embeddings', run_timer = run_timers)
    
    start = time.time()
    # Get distances from the embeddings
    df['distances'] = df["embeddings"].apply(lambda x: cdist([q_embeddings], [x], metric="Minkowski"))
    timer(start, func='Found embedding distances', run_timer = run_timers)

    start = time.time()
    returns = []
    cur_len = 0

    # Sort by distance and add rows to context until it is too long
    for i, row in df.sort_values("distances").iterrows():

        # Add length of tokens to the current length
        cur_len += row['n_tokens'] + 4

        # If current length is too long, break
        if cur_len > max_len:
            break

        # Add text to returns
        returns.append(row['text'])
        
    timer(start, func = "Retrieved nearest embeddings", run_timer = run_timers)

    return "\n\n###\n\n".join(returns)


def answer_question(question, df, model = "gpt-4o", max_len = 500, debug = False, max_tokens = 150, stop_sequence = None, rolling_context = ""):
    """Answer a question based on the most similar context from the dataframe texts"""

    # Create a context
    context = create_context(question, df, max_len = max_len)
    
    # System prompt   
    system_prompt = """You are Will, being spoken to over Whatsapp messages by Daisy. In the context, messages marked Daisy: [message] are from Daisy, and messages marked Will: [message] are from Will.
                        Answer as if you are Will and only Will based on the context provided and previous messages. Do not under any circumstances answer as Daisy. 
                        Be casual in the reply and keep it brief, but feel free to send multiple messages in one response separated by \n. Finish the message with a question back where appropriate to keep the conversation going."""

    # If debug, print the raw model no
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    
    # Get chat response
    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model = model, messages = [
                {"role": "system", "content": system_prompt},
                #{"role": "user", "content": f"Previous messages: {rolling_context}\n\n---\n\nContext: {context}\n\n---\n\nQuestion: {question}\n\n---\n\Will: "}
                {"role": "user", "content": f"\n\n---\n\nContext: {context}\n\n---\n\nQuestion: {rolling_context}'\n'{question}\n\n---\nWill: "}
            ],
            temperature = 0.5,
            max_tokens = max_tokens,
            stop = stop_sequence
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""


def run_program(chat_title):
    """Program to run chatbot until user quits"""
        
    # Load in df
    df = pd.read_pickle(f'./embeddings/embeddings_{chat_title}.pkl')

    # If embedding has not been converted to np array, do this now
    if type(df['embeddings'][0]) == 'str':
        print("Embeddings not saved as np array. Converting from string now...")
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        
    # Find chat members
    start = time.time()
    members = []
    for message in df['text']:
        for line in message.split('\n'):
            match = re.match(r'^([A-Za-z]+\s?[A-Za-z]*):', line)
            if match is not None:
                member = match.group(1)
                if member not in members:
                    members.append(member)
    timer(start, "Found members")
    print(members)
        
        
    # Preset rolling context to blank string
    rolling_context = ""
    
    while True:
    
        # Request user prompt
        prompt = input("Please enter a message from Daisy (R to reset, X to quit): ")
        
        # Check for quit or reset
        if prompt.lower() == 'x':
            break
        elif prompt.lower() == 'r':
            rolling_context = ""
            continue
        
        # Get chat response
        output = answer_question("Daisy: " + prompt, df, rolling_context = rolling_context)
        
        # Format new lines properly
        output = clean_output(output)
        
        # Combine prompt and output into a string and print
        previous_msgs = "\n".join([f"Daisy: {prompt}", f"{output}"])
        
        # Make generated conversation
        print(f"Generated conversation:\n{previous_msgs}")
        
        # Create remembered context
        if rolling_context != "":
            rolling_context += '\n'    
        rolling_context = previous_msgs
    

run_program('daisy')