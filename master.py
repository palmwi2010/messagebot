from whatsapp_bot import WhatsAppBot
import pandas as pd

# Set parameters and import embeddings df
chat_label = "Daisy"
msg_initial = True
loop_num = 0
df = pd.read_pickle('embeddings_check_embed.pkl')

# Initialize whatsapp bot
bot = WhatsAppBot(chat_label = chat_label, df=df)

# Initalize whatsapp connection
bot.initialize_driver()

# Start endless loop
while loop_num < 10:

    # Wait for new messages unless it is set to message immediately
    if not msg_initial or loop_num > 0:
        bot.check_new_messages()

    # Get current messages
    current_msgs = bot.get_messages()
    bot.messages = current_msgs

    # Generate messages
    new_msgs = bot.generate_messages(current_msgs)

    # Submit messages
    bot.submit_messages(messages=new_msgs, submit=True)
    
    # Increment loop number
    loop_num += 1