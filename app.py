import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from threading import Thread
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from doc_loader import get_table_data, async_html_loader, load_json
import os
from link_extractor import crawl
from dotenv import load_dotenv
import asyncio
from agent_tools import agent
import argparse
import time


load_dotenv()

SYSTEM_MESSAGE = """
Eres un educado y carismático asistente de una tienda virtual llamada Conair México que vende herramientas de estilismo. 
Los usuarios te realizarán consultas sobre los productos y demás información que yace en la página web www.conairmexico.com. 
Debes de tener en cuenta las siguientes consideraciones: 
    1. Algunos productos parecen tener dos precios, sin embargo, el primero es el precio sin descuento y el segunto es el precio final
    con descuento. Ejemplo: 
        QUESTION: ¿ cuál es el precio de la alaciadora Infiniti Pro 2 en 1 ?
        ANSWER: El precio es de $1699.00, pero con descuento $1274.25
    2. Responde con alegría y con tono persuasivo con el cliente.
    3. Responde de forma breve y concisa.
    4. Pregunta amablemente si deseean hacer otra consulta.
"""

def get_args():
    parser = argparse.ArgumentParser(description='Argument for running crawling')
    parser.add_argument("--use_crawling", action='store_true',
                        help="Runs again crawling for whole website. It might take a few minutes.")
    
    return parser.parse_args()


class ChatbotApp:
    def __init__(self, root, use_crawling):
        self.root = root
        self.root.title("Conair México Chatbot")

        self.use_crawling = use_crawling
    
        self.chat_history = []

        self.initial_greeting = """¡Hola! Bienvenido a Conair México. Estoy aquí para ayudarte con 
            cualquier consulta que tengas sobre nuestros productos o servicios. ¿En qué puedo asistirte hoy?"""

        self.chat_history.append(AIMessage(content=self.initial_greeting))

        thread = Thread(target=self.setup_agent_and_vector_stores(use_crawling=self.use_crawling))
        thread.start()
        self.show_message()

        self.setup_ui()

    def show_message(self):
        self.message_label = tk.Label(self.root, text="Bienvenido a Conair México Chatbot", font=("Helvetica", 16))
        self.message_label.pack(pady=20)
        self.message_label.after(10000, self.message_label.destroy)

    def setup_ui(self):
        self.chat_display = scrolledtext.ScrolledText(self.root, state='disabled', width=80, height=20)
        self.chat_display.pack(padx=10, pady=10)

        self.chat_display.tag_configure("assistant", foreground="yellow", font=("Helvetica", 12, "bold"))
        self.chat_display.tag_configure("user", foreground="green", font=("Helvetica", 12, "bold"))

        self.user_input = tk.Entry(self.root, width=70)
        self.user_input.pack(side=tk.LEFT, padx=(10, 0), pady=(0, 10))

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=10, pady=(0, 10))

        self.display_message(f"ASSISTANT: \t {self.initial_greeting}", "assistant")

    
    def setup_agent_and_vector_stores(self, use_crawling):

        if use_crawling:
            url = "https://www.conairmexico.com/"
            start_time = time.time()
            self.all_links = asyncio.run(crawl(url))
            final_time = time.time() - start_time
            print(f"Time in crawling : {final_time}")
            print(f"Length of pages: {len(self.all_links)}")
        else:
            self.all_links = load_json()

        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", 
                             temperature=0, api_key=os.getenv("OPENAI_API_KEY"),
                             max_tokens=200)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.__contact_url = "https://www.conairmexico.com/contact-us.html"
        self.__documents = async_html_loader(self.all_links)
        self.__all_tables = get_table_data(self.__contact_url)
        tables_doc = Document(page_content=str(self.__all_tables), metadata={"source": self.__contact_url})

        ## SPLITTER
        print("SPLITTING DOCUMENTS...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        splits = splitter.split_documents(self.__documents)

        ## INDEXING: EMBEDDINGS AND VECTOR STORE
        print("INDEXING...")
        embeddings = OpenAIEmbeddings()
        self.__vector_store_gral = FAISS.from_documents(splits, embedding=embeddings)
        self.__vector_store_tables = FAISS.from_documents([tables_doc], embedding=embeddings)
        self.vector_stores = [self.__vector_store_gral, self.__vector_store_tables]

        self.__retriever_gral = self.vector_stores[0].as_retriever(search_kwargs={"k": 3})
        self.__retriever_tables = self.vector_stores[1].as_retriever(search_kwargs={"k": 2})

        self.retrievers = [self.__retriever_gral, self.__retriever_tables]

        self.agent_executor = agent(self.retrievers, self.llm, self.prompt)

    def send_message(self):
        user_message = self.user_input.get()
        if user_message:
            self.chat_history.append(HumanMessage(content=user_message))
            self.display_message(f"USER: \t {user_message}", "user")
            self.user_input.delete(0, tk.END)
            self.process_and_display_response()

    def process_and_display_response(self):
        def run_chat():
            try:
                response = self.agent_executor.invoke({
                    "input": self.user_input.get(),
                    "chat_history": self.chat_history
                })["output"]
                self.chat_history.append(AIMessage(content=response))
                self.display_message(f"ASSISTANT: \t {response}", "assistant")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        thread = Thread(target=run_chat)
        thread.start()

    def display_message(self, message, message_type):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{message}\n\n", message_type)
        self.chat_display.configure(state='disabled')
        self.chat_display.yview(tk.END)

if __name__ == "__main__":

    args = get_args()

    root = tk.Tk()
    app = ChatbotApp(root, use_crawling=args.use_crawling)
    root.mainloop()
