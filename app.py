from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast

import chainlit as cl
import os

from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient 

api_key = 'insert your key here'

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = api_key
)

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="my_documents")
#db1 = Qdrant(client=client, embeddings=embeddings, collection_name="summary_db")

'''
def dict_to_comma_separated_string(d):
    return ', '.join(f"{key}: {value}" for key, value in d.items())

'''
'''
def compare_metadata(meta1, meta2):
    keys_to_compare = ["author", "publication_date", "title", "_collection_name"]
    for key in keys_to_compare:
        if meta1.get(key) != meta2.get(key):
            return False
    return True
'''
    
chat_history = []

@cl.on_chat_start
async def on_chat_start():

    print("A new chat session has started!")
    global chat_history
    runnable = llm | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    global chat_history
    global db
    #global db1

    if(message.content == '/clear'):
        chat_history = []
        await cl.Message(
            content=f"Chat has been cleared!",
        ).send()
        return
    
    query = message.content + str(chat_history).strip('[]')
    docs = db.similarity_search_with_score(query=query, k=5)
    best_chunks = ""

    for i in docs:
        doc, score = i
        print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
        best_chunks += doc.page_content + ', '

    initial_prompt = [
        (
            "system", 
            "You are a political ethics assistant who is given access to various works from philosophers."
        ),
        (
            "assistant", 
            "If the chunks given assists in any way to the question given you MUST use them. If you use the chunks you MUST write the phrase 'wow wow wow' at the beginning: " + best_chunks
        ),
    ]

    prompt = ChatPromptTemplate.from_messages(initial_prompt + chat_history + [("human", message.content)] )

    runnable = prompt | cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    chat_history.append(("human", message.content))
    chat_history.append(("assistant", msg.content))