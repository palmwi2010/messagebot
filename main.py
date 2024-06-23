"""Script to turn embeddings df into a response"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from openai import OpenAI
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
start = time.time()
API_KEY = os.getenv('API_KEY')
client = OpenAI(
    api_key = API_KEY
)
check = time.time()
print(f"Opened API connection in {check - start} seconds")

start = time.time()
df = pd.read_csv('embeddings.csv', index_col = 0)
print(f"Loaded df in {time.time() - start} seconds")

start=time.time()
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
print(f"Applied eval in {time.time() - start} seconds")


def create_context(question, df, max_len = 1000):
    """Create a context for a question by finding the most similar context from the dataframe"""

    start = time.time()
    # Get embeddings from the question
    q_embeddings = client.embeddings.create(input = question, model="text-embedding-3-large").data[0].embedding
    check = time.time()
    print(f"Created embeddings in {check - start} seconds")
    
    start = time.time()
    # Get distances from the embeddings
    df['distances'] = df["embeddings"].apply(lambda x: cdist([q_embeddings], [x], metric="Minkowski"))
    print(f"Calculate cdist in {time.time() - start} seconds")
    
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

    return "\n\n###\n\n".join(returns)


def answer_question(
    question,
    df,
    model = "gpt-3.5-turbo",
    max_len = 500,
    debug = False,
    max_tokens = 150,
    stop_sequence = None
):
    """Answer a question based on the most similar context from the dataframe texts"""

    # Create a context
    context = create_context(question, df, max_len = max_len)

    # If debug, print the raw model no
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    
    # Get chat response
    try:
        # Create a chat completion using the question and context
        rolling_context = ""
        response = client.chat.completions.create(
            model = model, messages = [
                {"role": "system", "content": "You are Will. Answer as if you are Will based only on the context provided and nothing else, and referring to the previous  messages. Be casual in the reply and keep it brief. Finish the message with a question back to keep the conversation going."},
                {"role": "user", "content": f"Previous messages: {rolling_context}\n\n---\n\nContext: {context}\n\n---\n\nQuestion: {question}\n\n---\n\Will: "}
            ],
            temperature = 0.8,
            max_tokens = max_tokens,
            stop = stop_sequence
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""


QUESTION = "Is that your willbot?"
output = answer_question(QUESTION, df, debug=False)
print(f"Question: {QUESTION}")
print(f"Answer: {output}")