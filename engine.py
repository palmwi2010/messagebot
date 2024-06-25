"""Class to turn embeddings df into a response"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from openai import OpenAI
import time
from utils import timer, clean_output
import os
from dotenv import load_dotenv
import re

class ChatEngine():
    def __init__(self, chat_title, mood = ""):
        # Load environment variables
        load_dotenv()
        
        # Config
        self.chat_title = chat_title
        self.responder = os.getenv('RESPONDER')
        self.api_key = os.getenv('API_KEY')
        self.client = OpenAI(api_key = self.api_key)
        self.mood = mood
        
        # Load in df
        try:
            fname = f'./embeddings/embeddings_{chat_title}.pkl'
            self.df = pd.read_pickle(fname)
        except:
            raise Exception(f'Could not open file {fname}')

        # If embedding has not been converted to np array, do this now
        if type(self.df['embeddings'][0]) == 'str':
            print("Embeddings not saved as np array. Converting from string now...")
            self.df['embeddings'] = self.df['embeddings'].apply(eval).apply(np.array)
        
        # Set members and system prompt
        self.members = self.get_chat_members()
        self.system_prompt = self.create_system_prompt()
        
        # Set root conversation history (to be able to reset it later) and a rolling conversation history
        self.root_conversation_history = [{"role": "system", "content": self.system_prompt}]
        self.conversation_history = self.root_conversation_history
            

    def create_system_prompt(self):
        """Function to create the system prompt for the API"""
        
        # Get members 
        members_str = ', '.join(self.members)
        
        return f"""You are {self.responder}, speaking in a Whatsapp chat with {members_str}. Your mood is extremely {self.mood}, and keep that mood throughout. In the context, messages marked member: [message] are from the given member, and messages marked {self.responder}: [message] are from you.
        Answer as if you are {self.responder} and only {self.responder} based on the context provided and previous messages. Do not under any circumstances answer as anyone else. Stick to the current conversation topics and avoid random context shifts.
        Be casual in the reply and keep it brief, but feel free to send multiple messages in one response separated by \n. Finish the message with a question back where appropriate to keep the conversation going."""

        
    def get_chat_members(self):
        """Function to get all members excluding responder in the whatsapp chat"""
        
        members = []
        for message in self.df['text']:
            for line in message.split('\n'):
                match = re.match(r'^([A-Za-z]+\s?[A-Za-z]*):', line)
                if match is not None:
                    member = match.group(1)
                    if member not in members and member != self.responder:
                        members.append(member)
        return members
    
    def create_context(self, question, max_tokens=500):
        """Function to create the context for the model based on the question asked"""
        
        # Create the question embeddings and calculate distances to them
        q_embeddings = self.client.embeddings.create(input=question, model="text-embedding-3-large").data[0].embedding
        self.df['distances'] = self.df["embeddings"].apply(lambda x: cdist([q_embeddings], [x], metric="Minkowski"))

        # Preallocate conversations, current len and the counter
        conversations, cur_len = [], 0

        # Loop through all rows and add to the conversations if it's not already too long
        for _, row in self.df.sort_values("distances").iterrows():
            cur_len += row['n_tokens'] + 4
            if cur_len > max_tokens:
                break
            conversations.append(row['text'])
        
        # Separate conversations with two new lines
        return "\n\n".join(conversations)
    
    
    def answer_question(self, question, model = "gpt-4o", max_context_tokens = 500, debug = False, max_output = 300):
        """Answer a question based on the most similar context from the dataframe texts"""

        # Create a context
        context = self.create_context(question, max_tokens = max_context_tokens)
        
        # Create new question_context line
        question_context = [{"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\n\n---\n\{self.responder}: "}]
        messages = self.conversation_history + question_context
        print(f"MESSAGES:\n{messages}")

        # If debug, print the raw model no
        if debug:
            print("Context:\n" + context)
            print("\n\n")
        
        # Get chat response
        try:
            # Create a chat completion using the question and context
            response = self.client.chat.completions.create(
                model = model, messages = messages, temperature = 0.5, max_tokens = max_output)
            
            # Get answer
            answer = clean_output(response.choices[0].message.content)
            
            # Add question and response to conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({'role': 'assistant', "content": answer})
            
            return answer
        except Exception as e:
            print(e)
            return ""
        
    
    def reset_conversation_history(self):
        """Function to reset the conversation history so far"""
        return [{"role": "system", "content": self.system_prompt}]
        
        
    def run(self):
        """Run chatbot with a command line interface"""
            
        # Loop in perpetuity
        while True:
        
            # If there's more than one other member, set who to speak
            if len(self.members) > 1:
                sender = ""
                while sender not in self.members:
                    sender = input(f"Please select who you are sending a message from? Options are {', '.join(self.members)}: ")
            else:
                sender = self.members[0]
        
            # Request user prompt
            prompt = input(f"Please enter a message from {sender} (R to reset, X to quit): ")
            
            # Check for quit or reset
            if prompt.lower() == 'x':
                break
            elif prompt.lower() == 'r':
                self.conversation_history = self.root_conversation_history
                continue
            
            # Get chat response
            question = f"{sender}: {prompt}"
            output = self.answer_question(question)
            
            # Create previous msgs string
            previous_msgs = [row['content'] for row in self.conversation_history if row['role'] != 'system']
            previous_msgs = '\n'.join(previous_msgs)
            
            # Make generated conversation
            print(f"Generated conversation:\n{previous_msgs}")


# Start engine
engine = ChatEngine('Daisy', 'funny')

# Run engine
engine.run()