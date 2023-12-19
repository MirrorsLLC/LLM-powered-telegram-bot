
import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import openai

# Set your OpenAI API key here
openai.api_key = 'your-openai-api-key'

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# Dictionary to store conversation history for each user
conversation_history = {}

# Function to generate a response using OpenAI's GPT-3.5-turbo
async def generate_response(user_id, user_message):
    # Initialize conversation history for the user if not present
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Append user's message to the conversation history
    conversation_history[user_id].append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the appropriate model identifier
            messages=conversation_history[user_id]
        )
        # Extract the response
        bot_response = response.choices[0].message['content']
        # Append bot's response to the conversation history
        conversation_history[user_id].append({"role": "assistant", "content": bot_response})

        return bot_response
    except Exception as e:
        logger.error(f"Error in generating response: {e}")
        return "Sorry, I couldn't process that message."

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Hello! I am your joke-telling assistant.')

async def bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_input = update.message.text
    user_id = update.message.from_user.id
    bot_response = await generate_response(user_id, user_input)
    await update.message.reply_text(bot_response)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Goodbye!')

def main() -> None:
    application = Application.builder().token('your-telegram-bot-token').build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_reply))
    application.add_handler(CommandHandler("cancel", cancel))

    application.run_polling()

if __name__ == '__main__':
    main()
