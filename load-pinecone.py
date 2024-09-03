import getpass
import os
import time
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



# if not os.getenv("PINECONE_API_KEY"):
#     os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

# pinecone_api_key = os.environ.get("PINECONE_API_KEY")
# os.environ["OPENAI_API_KEY"] = getpass.getpass()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get('SLACK_APP_TOKEN')
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


embeddings = OpenAIEmbeddings()
index_name='tj-slack'
vector_store = PineconeVectorStore(index=index_name, embedding=embeddings)

loader = PyPDFLoader(
    "Onboarding.pdf",
)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

vectorstore_from_docs = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddings
)

