from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langsmith import traceable

import os
from dotenv import load_dotenv
from pprint import pprint
import json

load_dotenv()

### TEMPLATE 

# TEMPLATE = """
# Eres un educado y amable asistente de un médico que lo ayuda a dar datos del historial clínico de un paciente, e incluso sugirere medicamentos o diagnósticos para el paciente.

# Sigue los siguientes principios para asistir al médico:
#     1. Debes saludar cordialmente al médico por su nombre {dr_name}, y preguntár en qué puedes ayudarlo o asistirlo.
#     2. Eres únicamente un asistente, por lo tanto no tienes la verdad absoluta sobre qué medicamentos puede tomar el paciente o el diagnóstico del mismo.
#     3. Si no estás seguro a qué paciente se refiere el médico, pregunta el nombre completo o más datos como la fecha de nacimiento del paciente.
#     4. Cuando el médico haya dado las gracias o se despida, debes preguntar si hay algo más en lo que lo puedas ayudar mencionando su nombre {dr_name}, 
#     de lo contrario también debes despedirte cordialmente.
# Context: {context}
# Question: {input}
# """

SYSTEM_MESSAGE = """
Eres un educado y amable asistente del médico {dr_name}, el cual lo ayuda a dar datos del historial clínico del paciente {patient_name}, 
e incluso sugirere medicamentos o diagnósticos para el paciente.

Además, debes seguir los siguientes principios para asistir al médico:
    1. Debes saludar cordialmente al médico por su nombre ({dr_name}), y preguntár en qué puedes ayudarlo o asistirlo.
    2. Eres únicamente un asistente, por lo tanto no tienes la verdad absoluta sobre qué medicamentos puede tomar el paciente o el diagnóstico del mismo.
    3. Cuando el médico haya dado las gracias o se despida, debes preguntar si hay algo más en lo que lo puedas ayudar mencionando su nombre {dr_name}, 
    de lo contrario también debes despedirte cordialmente.
"""

@traceable
def get_document_from_json(file_path):
    ## LOAD DATA
    loader = JSONLoader(file_path,
                        jq_schema=".",
                        text_content=False)

    json_data = loader.load()

    json_dict = json.loads(json_data[0].page_content)
    patient_name = json_dict['nombre']

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, 
                                            chunk_overlap=20)

    documents = splitter.split_documents(json_data)

    return documents, patient_name


def create_vector(docs):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embedding=embeddings)

    return vector_store


def agents_and_retriever(retriever, llm, prompt):
    retriever_tool = create_retriever_tool(
        retriever,
        "retriever_search",
        "Use this tool when searchinf for information about the patient"
        )
    med_tool = PubmedQueryRun()
    tools = [med_tool, retriever_tool]

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


def process_chat(vector_store, user_input, chat_history, patient_name):

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", 
                     temperature=0, api_key=os.getenv("OPENAI_API_KEY"),
                     max_tokens=200)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    agent_executor = agents_and_retriever(retriever, llm, prompt)

    response = agent_executor.invoke({
        "dr_name": "Dr. Jesús López",
        "patient_name": patient_name,
        "input": user_input,
        "chat_history": chat_history
    })

    return response["output"]


# if __name__ == '__main__':
#     file_path = "data/paciente1.json"
#     documents, patient_name = get_document_from_json(file_path)
#     vector_store = create_vector(documents)
#     # chain = create_chain(vector_store)

#     chat_history = []

#     while True:
#         user_input = input("MEDICO: \t")

#         if user_input.lower() == 'exit':
#             break

#         response = process_chat(vector_store, user_input, chat_history, patient_name)
#         chat_history.append(HumanMessage(content=user_input))
#         chat_history.append(AIMessage(content=response))
#         print()
#         print("ASISTENTE: \t", response)
#         print()