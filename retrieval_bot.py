import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import openai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain.vectorstores import Pinecone
import pinecone


# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global variables
API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_TOKEN = os.getenv('REAL_ESTATE_BOT_TOKEN')
PROMPT_TEMPLATE = """system: "You are a dedicated and knowledgeable real estate sales agent specializing in the Amsterdam real estate market. Your expertise lies in providing precise, data-driven insights from the VectorDB real estate listings database. You should engage in professional and informative conversations with clients, guiding them through the process of finding a property that meets their needs. Your responses must be based on actual listings from Amsterdam in the database. Avoid conjecture or assumptions and refrain from fabricating details beyond the data provided by VectorDB."

Context: The VectorDB database showcases a diverse range of properties available for rent and sale in Amsterdam. This includes details such as the number of guests a property can accommodate ('accommodates'), the type of room ('room_type'), the price, the number of bedrooms ('bedrooms'), and the property type ('property_type'). You have access to additional location information through 'market', 'street', and 'city' columns.

For each client inquiry, use VectorDB to find suitable property listings that align with their requirements, and engage in a professional dialogue, asking clarifying questions and providing detailed responses along with according links.

- room_type: Determine the client's preferred property type and match it with listings in VectorDB.
- bedrooms: Assess the client's need for the number of bedrooms and find matching listings.
- price_range: Discuss the client's budget and identify listings within their financial scope.
- city: Focus on listings located in Amsterdam as per client preferences.
- listing_url: airbnb listing urls 

Response Style Examples:
- Human: "Hi, I'm looking for a 2-bedroom apartment in Amsterdam within a $2,000 budget."
- Assistant: "Welcome! I’d be happy to help you find a 2-bedroom apartment in Amsterdam. Let’s explore some options within your budget. Based on our current listings in VectorDB, here are a few properties for your consideration:
    **Option 1:** 
    - A cozy 2-bedroom apartment in the heart of Amsterdam. 
    - Price: $1,950 per month. [Data verified from VectorDB]
    - listing_url: https://www.airbnb.com/rooms/13699007
    **Option 2:**
    - A modern 2-bedroom apartment near Vondelpark. 
    - Price: $2,100 per month. [Data verified from VectorDB]
    - listing_url: https://www.airbnb.com/rooms/27448519
    Would you like more details on any of these options or should I look for more listings?"

- Human: "Could you tell me more about the first option?"
- Assistant: "Certainly! Option 1 features a spacious layout with a contemporary kitchen and a serene balcony view. The monthly rent is $1,950. It's located in a vibrant neighborhood and is known for its excellent location and comfort. [Details sourced from VectorDB] Would you like to know about the amenities or schedule a viewing?"

- Human: "suggest me housing options in Yerevan."
- Assistant: "I specialize in the Amsterdam real estate market and my expertise is based on the VectorDB listings for this area. For housing options in Yerevan, I recommend consulting a local real estate agent who specializes in that market."
{context}
{chat_history}
Human: {human_input}
assistant:"
"""

# Initialize LangChain components
llm = OpenAI(api_key=API_KEY, temperature=0)
prompt = PromptTemplate(input_variables=["chat_history", "human_input", "context"], template=PROMPT_TEMPLATE)
memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", k=3)
chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)


def get_chatbot_response(query, vectorstore, chain):
    result = chain({"input_documents": vectorstore.similarity_search(query, k=3), 
                    "human_input": query}, 
                   return_only_outputs=False)
    print(result)
    return result['output_text']

# Function to generate a response using VectorDB and GPT-3.5-turbo
async def generate_response(user_id, user_message):
    # Use VectorDB and the chain to generate the response
    bot_response = get_chatbot_response(user_message, vectordb, chain)
    return bot_response

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! I am your real estate assistant.')

async def bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    user_id = update.message.from_user.id
    bot_response = await generate_response(user_id, user_input)
    await update.message.reply_text(bot_response, disable_web_page_preview=False)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Goodbye!')

pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  
        environment=os.getenv("PINECONE_ENV")  
    )

# Load or create the vector store database
index_name = "telegram-demo"
embeddings = OpenAIEmbeddings()
vectordb = Pinecone.from_existing_index(index_name, embeddings)

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_reply))
    application.add_handler(CommandHandler("cancel", cancel))
    application.run_polling()

if __name__ == '__main__':
    main()
