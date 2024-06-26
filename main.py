from whatsapp_bot import WhatsAppBot
from engine import ChatEngine

# Set whether to run chatbot in WhatsApp or using command line interface
cli_mode = True

# Set chat title, mood and whether to message immediately on startup
chat_label = "Daisy"
mood = "funny"

if cli_mode:
    # Initialize and run engine
    engine = ChatEngine(chat_label, mood=mood)
    engine.run()
else:

    # Set loop number, max responses and whether to message immediately
    loop_num = 0
    max_responses = 20
    msg_initial = True

    # Initialize whatsapp bot with chat label and mood
    bot = WhatsAppBot(chat_label = chat_label, mood = mood)

    # Initalize whatsapp connection
    bot.initialize_driver()

    # Start endless loop
    while loop_num < max_responses:

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