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
DB_PATH = '/home/armen/Documents/airbnb_demo/chroma_rbnb10000_few_features'
DATA_PATH = '/home/armen/Documents/miscellaneous/listing_sample_10000.csv'
METADATA_COLUMNS = ['room_type', 'street', 'property_type', 'price', 'bedrooms']
DOCUMENT_COLUMNS = ['accommodates',
 'review_scores_checkin',
 'reviews_per_month',
 'listing_url',
 'review_scores_accuracy',
 'room_type',
 'market',
 'street',
 'property_type',
 'review_scores_communication',
 'cleaning_fee',
 'host_is_superhost',
 'security_deposit',
 'review_scores_value',
 'first_review',
 'id',
 'host_response_rate',
 'review_scores_location',
 'maximum_nights',
 'weekly_price',
 'price',
 'review_scores_rating',
 'bedrooms',
 'review_scores_cleanliness',
 'neighbourhood',
 'host_about',
 'number_of_reviews',
 'monthly_price',
 'host_response_time']
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


# Function to create documents for the vector store
def create_docs_for_db(data_input, metadata_columns: list, document_columns: list):
    """Creating Docs to store in vector db using either a CSV file or a pandas DataFrame.

    Args:
        data_input (Union[str, pd.DataFrame]): Path to the CSV file or a pandas DataFrame.
        metadata_columns (list): List of column names to include in the metadata.
        document_columns (list): List of column names to include in the content.

    Returns:
        list: List of Document objects.
    """

    # Helper function to convert numpy types to python native types
    def numpy_to_python(value):
        if isinstance(value, (np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.float64, np.float32)):
            return float(value)
        else:
            return value

    # Load data if a path is provided, otherwise assume it's a dataframe
    if isinstance(data_input, str):
        data = pd.read_csv(data_input)
    elif isinstance(data_input, pd.DataFrame):
        data = data_input
    else:
        raise ValueError("Invalid input type. Expected a pandas DataFrame or a path to a CSV file.")

    docs = []

    for i in range(data.shape[0]):
        raw = data.iloc[i]

        # Fetching content for each column to be included in the primary searchable content
        page_content = ', '.join([f"{col}: {raw[col]}" for col in document_columns if col in data.columns])

        # Creating metadata dictionary by converting numpy types to python types
        raw_metadata = {col: numpy_to_python(raw[col]) for col in metadata_columns if col in data.columns}

        doc = Document(page_content=page_content, metadata=raw_metadata)
        docs.append(doc)

    return docs

# Function to load or create the vector database
def get_chroma_db(db_path: str, data_path_or_dataframe, metadata_columns, document_columns):
    """Load an existing Chroma vector store or create a new one."""
    embedding = OpenAIEmbeddings()

    # Check if the vector database exists
    if not os.path.exists(db_path):
        print("Vector database not found, creating new database...")
        # Generate the documents since the vector store does not exist
        docs = create_docs_for_db(data_path_or_dataframe, metadata_columns, document_columns)
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=db_path
        )
        vectordb.persist()
    else:
        print("Loading existing vector database...")
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=db_path
        )

    return vectordb# Function to get the chatbot response

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
        api_key=os.getenv("PINECONE_API_KEY"),  # Replace with your API key
        environment=os.getenv("PINECONE_ENV")  # Replace with your environment
    )


index_name = "telegram-demo"
embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(index_name, embeddings)
# Load or create the vector store database
vectordb = docsearch#get_chroma_db(DB_PATH, DATA_PATH, METADATA_COLUMNS, DOCUMENT_COLUMNS)

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_reply))
    application.add_handler(CommandHandler("cancel", cancel))
    application.run_polling()

if __name__ == '__main__':
    main()
