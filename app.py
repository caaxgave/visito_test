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
from doc_loader import load_json
import os
from link_extractor import crawl
from dotenv import load_dotenv
import asyncio
from agent_tools import agent, create_vector_store, create_retrievers
import argparse
import time

#TODO: Reviwe and clean

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


def get_args():
    parser = argparse.ArgumentParser(description='Argument for running from crawling or from indexing')
    parser.add_argument("--run_from", default='chat', type=str,
                        help="""if 'crawling' it runs all the process from 
                        crawling>>loading files>>indexing>>chat. if 'indexing' it 
                        loads files and chat. If 'chat' it is ready to be used.""")
    
    return parser.parse_args()


class ChatbotApp:
    def __init__(self, root, run_from, initial_greeting):
        self.root = root
        self.root.title("Conair México Chatbot")
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        window_width = 800
        window_height = 600

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)

        root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

        self.run_from = run_from
    
        self.chat_history = []

        self.initial_greeting = initial_greeting

        self.chat_history.append(AIMessage(content=self.initial_greeting))

        self.setup_agent_and_vector_stores(run_from=self.run_from)

        self.setup_ui()


    def setup_ui(self):
        self.chat_display = scrolledtext.ScrolledText(self.root, state='disabled', width=120, height=40)
        self.chat_display.pack(padx=10, pady=10)

        self.chat_display.tag_configure("assistant", foreground="yellow", font=("Helvetica", 12, "bold"))
        self.chat_display.tag_configure("user", foreground="green", font=("Helvetica", 12, "bold"))

        self.user_input = tk.Entry(self.root, width=70)
        self.user_input.pack(side=tk.LEFT, padx=(10, 0), pady=(0, 10))

        self.user_input.bind('<Return>', self.on_enter_pressed)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=10, pady=(0, 10))

        self.display_message(f"ASSISTANT: \t {self.initial_greeting}", "assistant")

    
    def on_enter_pressed(self, event):
        self.send_message()


    def setup_agent_and_vector_stores(self, run_from):

        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", 
                             temperature=0, api_key=os.getenv("OPENAI_API_KEY"),
                             max_tokens=200)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        if run_from == 'crawling':
            url = "https://www.conairmexico.com/"

            self.all_links = asyncio.run(crawl(url))
            self.vector_stores = create_vector_store(self.all_links)

            self.retrievers = create_retrievers(vector_stores=self.vector_stores)
            self.agent_executor = agent(self.retrievers, self.llm, self.prompt)

        elif run_from == 'indexing':
            self.all_links = load_json()
            self.vector_stores = create_vector_store(self.all_links)
            self.retrievers = create_retrievers(vector_stores=self.vector_stores)
            self.agent_executor = agent(self.retrievers, self.llm, self.prompt)
        else:
            embeddings = OpenAIEmbeddings()
            gral_index = FAISS.load_local("Indexes/gral_index/", embeddings, allow_dangerous_deserialization=True)
            tables_index = FAISS.load_local("Indexes/table_index/", embeddings, allow_dangerous_deserialization=True)
            self.vector_stores = [gral_index, tables_index]
            self.retrievers = create_retrievers(vector_stores=self.vector_stores)
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
    initial_greeting = """¡Hola! Bienvenido a Conair México. Estoy aquí para ayudarte con 
    cualquier consulta que tengas sobre nuestros productos o servicios. ¿En qué puedo asistirte hoy?"""

    root = tk.Tk()
    app = ChatbotApp(root, run_from=args.run_from, initial_greeting=initial_greeting)
    root.mainloop()
