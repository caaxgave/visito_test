from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
import argparse
from pprint import pprint
from link_extractor import crawl
import time
from tqdm import tqdm
import asyncio
import json
from langsmith import traceable
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def get_args():
    parser = argparse.ArgumentParser(description='Langchain documents for webpages in the list')
    parser.add_argument("--use_crawling", action='store_true',
                        help="Runs again crawling for whole website. It might take a few minutes.")
    
    return parser.parse_args()


def load_json():
    print("LOADING PAGES...")
    with open("all_valid_urls.json", "r") as file:
        loaded_list = json.load(file)

    return loaded_list


@traceable
def async_html_loader(all_links):
    loader = AsyncHtmlLoader(all_links)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    return docs_transformed


def get_table_data(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.content()

        # Parse the page content
        soup = BeautifulSoup(content, 'html.parser')
        tables = soup.find_all('table')

        if not tables:
            return None
        
        all_tables_data = {}

        for idx, table in enumerate(tables):
            table_data = {} 
            rows = table.find_all('tr')

            headers = [header.text.strip() for header in rows[0].find_all(['td', 'th'])] 

            j=1
            for row in rows[1:]:  
                cells = row.find_all(['td', 'th'])
                row_data = {headers[i]: cell.text.strip() for i, cell in enumerate(cells)}
                table_data[j] = row_data
                j+=1

            all_tables_data[f"Table_{idx+1}"] = table_data

        browser.close()

        return all_tables_data


if __name__ == '__main__':

    args = get_args()
    print(f"use_crawling is: {args.use_crawling}")

    if args.use_crawling:
        url = "https://www.conairmexico.com/"
        start_time = time.time()
        all_links = asyncio.run(crawl(url))
        final_time = time.time() - start_time
        print(f"Time in crawling : {final_time}")
        print(f"Length of pages: {len(all_links)}")
    else:
        all_links = load_json()

    contact_url = "https://www.conairmexico.com/contact-us.html"

    start_time = time.time()
    documents = async_html_loader(all_links)
    all_tables = get_table_data(contact_url)
    if all_tables:
        tables_doc = Document(page_content=str(all_tables), metadata={"source": contact_url})
        documents.append(tables_doc)
    final_time = time.time() - start_time

    pprint(documents)
    print(f"Time in loading docs: {final_time}")
    print(f"The length of Documents is: {len(documents)}")