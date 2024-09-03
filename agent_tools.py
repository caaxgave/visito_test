from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.vectorstores.faiss import FAISS
from doc_loader import get_table_data
from doc_loader import load_json, html_loader_product_data
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langsmith import traceable
import os
from dotenv import load_dotenv

#TODO: Review and clean

load_dotenv()

SYSTEM_MESSAGE = """
Eres un asistente virtual alegre, carismático, persuasivo y atento de la tienda en línea Conair México, especializada en herramientas 
de estilismo. Los usuarios te harán preguntas sobre productos y otra información disponible en la página web www.conairmexico.com. 
Debes responder de manera breve, concisa, y siempre ofrecer ayuda adicional al final de cada interacción.

Considera lo siguiente:
    1. Algunos productos muestran dos precios. El primero es el precio original y el segundo es el precio final con descuento. Ejemplo:
        PREGUNTA: ¿Cuál es el precio de la alaciadora Infiniti Pro 2 en 1?
        RESPUESTA: El precio original es de $1699.00, pero con el descuento es de $1274.25.
        
    2. Si el usuario pregunta por la disponibilidad de un producto, y hay stock, puedes mencionarlo y listar las opciones disponibles. 
       Si no hay stock, ofrece una disculpa y sugiere que el usuario revise más tarde o explore otros productos.
       
    3. Nunca inventes información. Si no encuentras la información solicitada de contacto, o el producto en la página web, 
    indica al usuario que lamentablemente no tienes esa información en este momento, y ofrece asistencia adicional.
       
    4. Mantén un tono profesional y cálido, asegurando que el usuario se sienta bien atendido. Si el usuario tiene más preguntas, 
       invita amablemente a que continúe preguntando.
"""


def create_vector_store(all_links):
    """
        Creates and returns vector stores for webpage content and table data.

        This function processes a list of webpage links, loads their content, extracts tables from 
        a contact page, and then splits and indexes the data into vector stores using embeddings.

        Args:
            all_links (list): A list of URLs to load and process.

        Returns:
            vector_stores (list): A list containing two FAISS vector stores; one for the webpage content and one 
                for the table data.
    """

    print("INDEXING...")

    contact_url = "https://www.conairmexico.com/contact-us.html"
    documents = html_loader_product_data(all_links)
    all_tables = get_table_data(contact_url)
    tables_doc = Document(page_content=str(all_tables), metadata={"source": contact_url})

    ## SPLITTER
    print("SPLITTING DOCUMENTS...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    splits = splitter.split_documents(documents)

    ## INDEXING: EMBEDDINGS AND VECTOR STORE
    embeddings = OpenAIEmbeddings()
    vector_store_gral = FAISS.from_documents(splits, embedding=embeddings)
    vector_store_tables = FAISS.from_documents([tables_doc], embedding=embeddings)
    vector_stores = [vector_store_gral, vector_store_tables]

    return vector_stores


def create_retrievers(vector_stores):
    """
        Creates retrievers from the provided vector stores.

        This function takes in a list of vector stores and converts them into retrievers 
        with specified search parameters.

        Args:
            vector_stores (list): A list of FAISS vector stores.

        Returns:
            retrievers (list): A list containing two retrievers; one for general content and one for table data.
    """

    retriever_gral = vector_stores[0].as_retriever(search_kwargs={"k": 2})
    retriever_tables = vector_stores[1].as_retriever(search_kwargs={"k": 2})
    retrievers = [retriever_gral, retriever_tables]

    return retrievers


@traceable
def agent(retrievers, llm, prompt):
    """
        Creates and returns an agent executor with specified retrievers and LLM.

        This function sets up tools for general product information and service center data 
        using the provided retrievers and integrates them into an agent with the given LLM and prompt.

        Args:
            retrievers (list): A list containing two retrievers; one for general content and one for table data.
            llm (object): The language model used by the agent.
            prompt (str): The prompt used to guide the agent's responses.

        Returns:
            agent_executor (object): An agent executor configured with the specified tools and prompt.
    """
        
    retriever_gral_tool = create_retriever_tool(
        retrievers[0],
        "general_retriever",
        """Utiliza esta herramienta cuando busque información sobre los productos, precios, ofertas,
        descripción del producto o información del número contacto o correo electrónico (email) de contacto de 
        Conair México en www.conairmexico.com."""
        )

    retriever_tables_tool = create_retriever_tool(
        retrievers[1],
        "service_center_and_workshops_retriever",
        """Utiliza esta herramienta cuando busque información sobre los centros de servicio o los talleres foráneos de
        Conair México en www.conairmexico.com."""
        )

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
    """
    Dummy function for test of chat processing interaction using a language model and vector stores.

    This function sets up the language model, prompt template, and retrievers. It then creates 
    an agent executor and uses it to handle the user input within the context of the chat history.

    Args:
        vector_stores (list): A list of vector stores used for retrieving relevant information.
        user_input (str): The user's message or query.
        chat_history (list): The history of previous messages in the chat.

    Returns:
        response (str): The agent's response to the user input.
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", 
                     temperature=0, api_key=os.getenv("OPENAI_API_KEY"),
                     max_tokens=200)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    retrievers = create_retrievers(vector_stores)

    agent_executor = agent(retrievers, llm, prompt)

    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    return response["output"]


if __name__ == '__main__':

    all_links = load_json()
    
    ## SCRAPPING ---> DOCUMENTS
    contact_url = "https://www.conairmexico.com/contact-us.html"

    documents = html_loader_product_data(all_links)

    with open("metadatas.txt", 'w') as f:
        for doc in documents:
            f.write(str(doc.metadata))
            f.write("\n")

    all_tables = get_table_data(contact_url)
    tables_doc = Document(page_content=str(all_tables), metadata={"source": contact_url})

    ## SPLITTER
    print("SPLITTING DOCUMENTS...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    splits = splitter.split_documents(documents)

    ## INDEXING: EMBEDDINGS AND VECTOR STORE
    print("INDEXING...")
    embeddings = OpenAIEmbeddings()

    ### FOR CREATING AND SAVING NEW VECTOR STORES
    vector_store_gral = FAISS.from_documents(splits, embedding=embeddings)
    vector_store_gral.save_local("Indexes/gral_index")
    vector_store_tables = FAISS.from_documents([tables_doc], embedding=embeddings)
    vector_store_tables.save_local("Indexes/table_index")

    ### FOR LOADING NORMAL INDEXES (AWARELESS BREADCRUMBS)
    gral_index = FAISS.load_local("Indexes/gral_index/", embeddings, allow_dangerous_deserialization=True)
    tables_index = FAISS.load_local("Indexes/table_index", embeddings, allow_dangerous_deserialization=True)

    vector_stores = [gral_index, tables_index]

    # CHAT
    chat_history = []

    initial_greeting = """¡Hola! Bienvenido a Conair México. Estoy aquí para ayudarte con 
    cualquier consulta que tengas sobre nuestros productos o servicios. ¿En qué puedo asistirte hoy?"""
    print(f"ASSISTANT: {initial_greeting}")
    chat_history.append(AIMessage(content=initial_greeting))

    print("CHAT IS READY...")
    while True:
        user_input = input("USER: ")
        answer = process_chat(vector_stores, user_input, chat_history)
        print(f"ASSISTANT: {answer}")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))