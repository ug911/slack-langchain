import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

class PineconeManager:
    def __init__(self, api_key, environment="us-east-1", embedding_dimension=1536):
        """
        Initializes the PineconeManager with API key and environment.

        :param api_key: Your Pinecone API key.
        :param environment: Pinecone environment (e.g., 'us-west1-gcp').
        :param embedding_dimension: Dimension of the embedding vector, default is 1536.
        """
        self.pinecone = Pinecone(api_key=api_key)
        self.embedding_dimension = embedding_dimension
        self.environment = environment
        self.index = None
        self.vectorstore = None

    def create_or_initialise_index(self, index_name):
        """
        Initializes a connection to an existing Pinecone index. Creates a Pinecone index if it does not exist.

        :param index_name: Name of the index to be created.
        """
        existing_indexes = [index_info["name"] for index_info in self.pinecone.list_indexes()]

        if index_name not in existing_indexes:
            self.pinecone.create_index(
                name=index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.environment),
            )
            while not self.pinecone.describe_index(index_name).status["ready"]:
                time.sleep(1)
            print(f"Index '{index_name}' created.")
        else:
            print(f"Index '{index_name}' already exists.")

        self.index = self.pinecone.Index(index_name)
        print(f"Index '{index_name}' initialized.")

    def init_vectorstore(self, index_name):
        """
        Defines the vectorstore using the initialized Pinecone index and OpenAI embeddings.

        :param index_name: Name of the index to be connected to.
        """
        if self.index is None:
            self.create_or_initialise_index(index_name=index_name)

        embeddings = OpenAIEmbeddings()
        self.vectorstore = PineconeVectorStore(index=self.index, embedding=embeddings)
        print(f"Vectorstore defined for index '{index_name}'.")

    def add_documents(self, documents, namespace=None):
        """
        Adds documents to the Pinecone index with optional namespace.

        :param documents: List of documents to add.
        :param namespace: Optional namespace for the documents.
        """
        if self.index is None:
            raise ValueError("Index is not initialized. Please initialize an index first.")

        if self.vectorstore is None:
            raise ValueError("VectorStore is not initialized. Please initialize your VectorStore first.")

        self.vectorstore.add_documents(documents, namespace=namespace)
        print(f"Added documents to index '{self.index.name}' under namespace '{namespace}'.")

    def remove_documents(self, ids, namespace=None):
        """
        Removes documents from the Pinecone index by IDs.

        :param ids: List of document IDs to remove.
        :param namespace: Optional namespace for the documents.
        """
        if self.index is None:
            raise ValueError("Index is not initialized. Please initialize an index first.")

        self.index.delete(ids=ids, namespace=namespace)
        print(f"Removed documents from index '{self.index.name}' under namespace '{namespace}'.")
