from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma
from utils import process_chat
from doc_loader import get_table_data
from doc_loader import load_json, async_html_loader
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
import time
from link_extractor import crawl
import argparse
from langsmith import traceable
import os
from dotenv import load_dotenv


load_dotenv()

SYSTEM_MESSAGE = """
Eres un educado y carismático asistente de una tienda virtual llamada Conair México que vende herramientas de estilismo. 
Los usuarios te realizarán consultas sobre los productos y demás información que yace en la página web www.conairmexico.com. 
Debes de tener en cuenta las siguientes consideraciones: 
    1. Algunos productos parecen tener dos precios, sin embargo, el primero es el precio sin descuento y el segunto es el precio final
    con descuento. Ejemplo: 
        QUESTION: ¿ cuál es el precio de la alaciadora Infiniti Pro 2 en 1 ?
        ANSWER: El precio es de $1699.00, pero con descuento $1274.25
    2. Contesta con alegría y se persuasivo con el cliente.
    3. Pregunta amablemente si deseean hacer otra consulta.
"""


def get_args():
    parser = argparse.ArgumentParser(description='Langchain documents for webpages in the list')
    parser.add_argument("--use_crawling", action='store_true',
                        help="Runs again crawling for whole website. It might take a few minutes.")
    
    return parser.parse_args()


@traceable
def agent(retrievers, llm, prompt):
    retriever_gral_tool = create_retriever_tool(
        retrievers[0],
        "general_retriever",
        """Utiliza esta herramienta cuando busque información sobre los productos, precios, ofertas,
        descripción del producto o información de contacto por teléfono o correo electrónico de 
        Conair México en www.conairmexico.com."""
        )

    retriever_tables_tool = create_retriever_tool(
        retrievers[1],
        "service_center_and_workshops_retriever",
        """Utiliza esta herramienta cuando busque información sobre los centros de servicio o los talleres foráneos de
        Conair México en www.conairmexico.com."""
        )

    #TODO: Create a new tool to be aware of product breadcrumbs.
    tools = [retriever_gral_tool, retriever_tables_tool]

    agent = create_openai_functions_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools
    )

    return agent_executor


def process_chat(vector_stores, user_input, chat_history):

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", 
                     temperature=0, api_key=os.getenv("OPENAI_API_KEY"),
                     max_tokens=200)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    retriever_gral = vector_stores[0].as_retriever(search_kwargs={"k": 3})
    retriever_tables = vector_stores[1].as_retriever(search_kwargs={"k": 2})

    retrievers = [retriever_gral, retriever_tables]

    agent_executor = agent(retrievers, llm, prompt)

    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    return response["output"]


if __name__ == '__main__':

    args = get_args()
    print(f"use_crawling is: {args.use_crawling}")

    ## CRAWLING
    if args.use_crawling:
        url = "https://www.conairmexico.com/"
        start_time = time.time()
        all_links = asyncio.run(crawl(url))
        final_time = time.time() - start_time
        print(f"Time in crawling : {final_time}")
        print(f"Length of pages: {len(all_links)}")
    else:
        all_links = load_json()
    
    ## SCRAPPING ---> DOCUMENTS
    print("LOADING DOCUMENTS...")
    contact_url = "https://www.conairmexico.com/contact-us.html"
    documents = async_html_loader(all_links)
    all_tables = get_table_data(contact_url)
    tables_doc = Document(page_content=str(all_tables), metadata={"source": contact_url})

    ## SPLITTER
    print("SPLITTING DOCUMENTS...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    splits = splitter.split_documents(documents)

    ## INDEXING: EMBEDDINGS AND VECTOR STORE
    print("INDEXING...")
    embeddings = OpenAIEmbeddings()
    vector_store_gral = FAISS.from_documents(splits, embedding=embeddings)
    vector_store_tables = FAISS.from_documents([tables_doc], embedding=embeddings)
    vector_stores = [vector_store_gral, vector_store_tables]
    # vector_store = Chroma.from_documents(splits, embedding=embeddings)

    ## CHAT
    chat_history = []
    print("CHAT IS READY...")

    initial_greeting = """¡Hola! Bienvenido a Conair México. Estoy aquí para ayudarte con 
    cualquier consulta que tengas sobre nuestros productos o servicios. ¿En qué puedo asistirte hoy?"""
    print(f"ASSISTANT: {initial_greeting}")
    chat_history.append(AIMessage(content=initial_greeting))

    while True:
        user_input = input("USER: ")
        answer = process_chat(vector_stores, user_input, chat_history)
        print(f"ASSISTANT: {answer}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))