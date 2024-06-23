from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import re
from engine import answer_question
from utils import delay, clean_output
import pandas as pd


class WhatsAppBot():
    def __init__(self, chat_label, df):
        self.df = df
        self.chat_label = chat_label
        self.messages = None
        self.editable_div = None
        self.prompt_row_length = 5
        self.msg_refresh_delay = 15
    
    
    def initialize_driver(self):
        """Function to establish WhatsApp connection and navigate to the messages page"""
        
        # Set connection to chrome and wait for input while user logs in
        self.driver = webdriver.Chrome()
        self.driver.get("https://web.whatsapp.com/")
        input('Press when ready to get page source')
        
        # Find and click the chat with user
        try:
            selenium_span = self.driver.find_element(By.XPATH, f"//span[text()='{self.chat_label}']")
        except:
            raise Exception(f"Unable to find {self.chat_label} in WhatsApp menu")
        selenium_span.click()
        self.delay_vs()
        
        # Save the editable div
        try:
            self.editable_div = self.driver.find_element(By.XPATH, "(//div[contains(@class, 'x1hx0egp') and @role='textbox'])[2]")
        except:
            raise Exception(f"Unable to find input text menu")
        
        
    def get_messages(self):
        """Function to get messages visible on WhatsApp screen, returning a row of dictionaries with sender, msg and gpt_msg (concatenation)"""
        
        # Prepare soup and message spans for finding messages
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        messages = soup.find_all(class_="_ao3e")
        
        # Assign messages and senders
        rows = []
        for msg in messages:
            # Content is one span beneath the class found
            msg_children = msg.findChildren("span")
            
            # Other same class exist so we need to ignore them
            if len(msg_children) > 0:
                
                # Assign to message content
                msg_content = msg_children[0].contents[0]
                
                # To find the sender, we go up 2 parents to a plan text property
                parent_div = msg.parent.parent
                data_pre_plain_text = parent_div.get('data-pre-plain-text')
                
                # Use re library to find just name
                match = re.search('([A-Za-z]+):', data_pre_plain_text)
                name = match.group(1)
                
                # Add message and sender to rows
                rows.append({'sender': name, 'msg': msg_content, 'gpt_msg': f"{name}: {msg_content}"})

        # Return the messages
        return rows
    
    
    def generate_messages(self, message_list):
        """Generate messages using OpenAI based on the previous messages in the list"""
        
        # Remove my most recent messages for ease of testing
        rows_test = message_list

        # Set the start index to avoid taking an index out of bounds
        start_idx = -1 * min(self.prompt_row_length, len(rows_test))

        # Loop back through from idx
        question_list = rows_test[start_idx:]
        gpt_msgs = [row['gpt_msg'] for row in question_list]
        prompt = '\n'.join(gpt_msgs)

        # Get chat response
        output = answer_question(prompt, self.df)

        # Format new lines properly
        output = clean_output(output)

        # Combine prompt and output into a string and print
        output_msgs = "\n".join([f"{prompt}", f"{output}"])

        # Print generated conversation
        print(f"Generated conversation:\n{output_msgs}")

        # Prepare messages to be sent
        output_array = output.split('\n')

        # Find just the message content
        output_clean = []
        for msg in output_array:
            
            # Remove whitespace and special characters
            msg = msg.strip()
            msg = ''.join(char for char in msg if ord(char) <= 0xFFFF)
            
            # Remove full stop and take only the message
            if msg[-1] == '.':
                msg = msg[:-1]
            match = re.search(r'^[A-Za-z]+: (.*)', msg)
            output_clean.append(match.group(1))
            
        return output_clean
    
    def submit_messages(self, messages, submit=False):
        """Function actually send messages on whatsapp"""
        for msg in messages:
            
            for char in msg:
                self.editable_div.send_keys(f'{char}')
                delay(0.02)
            
            if submit:
                self.editable_div.send_keys(Keys.ENTER)
            self.delay_vs()
            
    def check_new_messages(self):
        """Function that will run until a new message is received"""
        
        # Get current messages as a baseline
        if self.messages is None:
            current_messages = self.get_messages()
        else:
            current_messages = self.messages
            
        # Filter for only messages received
        print(current_messages[-1])
        current_messages_received = [row for row in current_messages if row['sender'] == 'Daisy']
        
        # Loop until new message sent
        while True:
            
            # Get latest messages
            latest_messages = self.get_messages()
            
            # Filter for only messages received
            latest_messages_received = [row for row in latest_messages if row['sender'] == 'Daisy']
            
            # Check if latest message is different
            if latest_messages_received[-1]['msg'] != current_messages_received[-1]['msg']:
                break
            else:
                self.delay_refresh()
                
        
    # Set delay functions
    def delay_vs(self):
        delay(0.5)

    def delay_s(self):
        delay(2)
        
    def delay_m(self):
        delay(5)
        
    def delay_l(self):
        delay(10)
        
    def delay_vl(self):
        delay(1000)
        
    def delay_refresh(self):
        delay(self.msg_refresh_delay)
"""         

def main():

    # Start driver and navigate to chrome
    driver = webdriver.Chrome()
    driver.get("https://web.whatsapp.com/")

    # Wait to sign in using whatsapp and then press to continue
    input('Press when ready to get page source:')

    # Find and click the chat with user
    selenium_span = driver.find_element(By.XPATH, f"//span[text()='{chat_label}']")
    selenium_span.click()
    delay(vs)

    # Locate the div element that we will use to send messages using XPath
    editable_div = driver.find_element(By.XPATH, "(//div[contains(@class, 'x1hx0egp') and @role='textbox'])[2]")

    # Prepare soup and message spans for finding messages
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    messages = soup.find_all(class_="_ao3e")

    # Assign messages and senders
    rows = []
    for msg in messages:
        # Content is one span beneath the class found
        msg_children = msg.findChildren("span")
        
        # Other same class exist so we need to ignore them
        if len(msg_children) > 0:
            
            # Assign to message content
            msg_content = msg_children[0].contents[0]
            
            # To find the sender, we go up 2 parents to a plan text property
            parent_div = msg.parent.parent
            data_pre_plain_text = parent_div.get('data-pre-plain-text')
            
            # Use re library to find just name
            match = re.search('([A-Za-z]+):', data_pre_plain_text)
            name = match.group(1)
            
            # Add message and sender to rows
            rows.append({'sender': name, 'msg': msg_content, 'gpt_msg': f"{name}: {msg_content}"})

    # Remove my most recent message for ease of testing
    rows_test = rows[:-1]

    # Set parameters for number of messages to include in context and question
    max_row_question = 5
    start_idx = -1 * min(max_row_question, len(rows_test))

    # Loop back through from idx
    question_list = rows_test[start_idx:]
    gpt_msgs = [row['gpt_msg'] for row in question_list]
    prompt = '\n'.join(gpt_msgs)

    # Get chat response
    output = answer_question(prompt, df)

    # Format new lines properly
    output = clean_output(output)

    # Combine prompt and output into a string and print
    output_msgs = "\n".join([f"{prompt}", f"{output}"])

    # Print generated conversation
    print(f"Generated conversation:\n{output_msgs}")

    # Prepare messages to be sent
    output_array = output.split('\n')

    # Find just the message content
    output_clean = []
    for msg in output_array:
        
        # Remove whitespace and special characters
        msg = msg.strip()
        msg = ''.join(char for char in msg if ord(char) <= 0xFFFF)
        
        # Remove full stop and take only the message
        if msg[-1] == '.':
            msg = msg[:-1]
        match = re.search(r'^[A-Za-z]+: (.*)', msg)
        output_clean.append(match.group(1))
        
    print(output_clean)
    for msg in output_clean:
        editable_div.send_keys(f'{msg}')
        editable_div.send_keys(Keys.ENTER)
        delay(m)

    
    delay(vl)
"""