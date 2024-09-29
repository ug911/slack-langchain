import asyncio
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.callbacks import AsyncCallbackManager

from slack_sdk import WebClient
from PineconeManager import PineconeManager
# How to do a search over docs with conversation:
#https://langchain.readthedocs.io/en/latest/modules/memory/examples/adding_memory_chain_multiple_inputs.html
# People talking about the parsing error: https://github.com/hwchase17/langchain/issues/1657

from AsyncStreamingSlackCallbackHandler import AsyncStreamingSlackCallbackHandler


DEFAULT_MODEL="gpt-4o-mini"
UPGRADE_MODEL="gpt-4o-mini"
DEFAULT_TEMPERATURE=1

class ConversationAI:
    def __init__(
        self, bot_name:str, slack_client:WebClient, pinecone_client: PineconeManager, existing_thread_history=None, model_name:str=None
    ):
        self.bot_name = bot_name
        self.existing_thread_history = existing_thread_history
        self.model_name = None
        self.agent = None
        self.model_temperature = None
        self.slack_client = slack_client
        self.pc = pinecone_client
        self.lock = asyncio.Lock()

    async def create_rag_agent(self, sender_user_info, initial_message):
        print('Inside create_rag_agent')
        self.model_name = DEFAULT_MODEL
        self.model_temperature = DEFAULT_TEMPERATURE
        # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.callbackHandler = AsyncStreamingSlackCallbackHandler(self.slack_client)

        llm1 = ChatOpenAI(model_name=self.model_name, temperature=self.model_temperature, request_timeout=60,
                          max_retries=3, streaming=True, verbose=True)

        llm2 = ChatOpenAI(model_name=self.model_name, temperature=self.model_temperature, request_timeout=60,
                         max_retries=3, streaming=True, verbose=True,
                         callback_manager=AsyncCallbackManager([self.callbackHandler]))

        # ### Construct retriever ###
        # loader = WebBaseLoader(
        #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        #     bs_kwargs=dict(
        #         parse_only=bs4.SoupStrainer(
        #             class_=("post-content", "post-title", "post-header")
        #         )
        #     ),
        # )
        # docs = loader.load()
        #
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # splits = text_splitter.split_documents(docs)
        # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        # retriever = vectorstore.as_retriever()

        retriever = self.pc.vectorstore.as_retriever(
            # search_type="mmr",
            # search_kwargs={'k': 1, 'fetch_k': 10}
        )
        ### Contextualize question ###
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm1, retriever, contextualize_q_prompt
        )

        ### Answer question ###
        qa_system_prompt = """You are an assistant made by engineers at Tech Japan (also Talendly). \
        You are supposed to send latest updates from documents which primarily has meeting notes and context about clients. \
        When asked about a company, do not give information about any other company even if it is in context. \
        Ignore any other company apart from the company being asked upon and do not talk about it. \
        If you are not confident about answer, respond politely that you do not know the answer. \
        Do not hallucinate and make up facts. \
        You have a delightful & helpful persona. You are never to abuse to talk bad about anyone or tech japan. \
        Refrain from talking about competitors. \

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm2, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        ### Statefully manage chat history ###
        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            print('Here with {}'.format(session_id))
            if session_id not in store:
                print('Not found with {}'.format(session_id))
                store[session_id] = ChatMessageHistory()
                existing_thread_history = self.existing_thread_history
                if existing_thread_history is not None:
                    for message in existing_thread_history:
                        sender_name = list(message.keys())[0]  # get the first key which is the name (assuming only one key per dictionary)
                        message_content = list(message.values())[0]  # get the first value which is the message content
                        if sender_name == "bot":
                            store[session_id].add_ai_message(message_content)
                        else:
                            store[session_id].add_user_message(message_content)
            print(store[session_id])
            print('-----------------------')
            return store[session_id]

        self.agent = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return self.agent

    async def get_or_create_rag_agent(self, sender_user_info, message):
        if self.agent is None:
            print('Did not find the agent')
            self.agent = await self.create_rag_agent(sender_user_info, message)
        return self.agent

    async def respond(self, sender_user_info, channel_id:str, thread_ts:str, message_being_responded_to_ts:str, message:str):
        async with self.lock:
          agent = await self.get_or_create_rag_agent(sender_user_info, message)
          print("Starting response...")
          await self.callbackHandler.start_new_response(channel_id, thread_ts)
          # Now that we have a handler set up, just telling it to predict is sufficient to get it to start streaming the message response...
          response = await self.agent.ainvoke({'input':message}, config={"configurable": {"session_id": thread_ts}})
          print(response)
          return response
