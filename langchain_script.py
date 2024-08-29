#RAG (Retrieval Augmented Generation)

from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter

import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")


def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["sender_name"] = record.get("sender_name")
    metadata["timestamp_ms"] = record.get("timestamp_ms")

    return metadata

loader = JSONLoader(file_path="data.json",
                    jq_schema=".",
                    text_content=False,
                    metadata_func=metadata_func)

json_data = loader.load()



splitter = RecursiveCharacterTextSplitter(chunk_size=200, 
                                          chunk_overlap=20)

documents = splitter.split_documents(json_data)

#texts = splitter.split_documents(documents)

pprint(documents)
pprint(len(documents))
#pprint(json_chunks[0])

