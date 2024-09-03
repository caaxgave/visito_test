from langchain_community.document_loaders import WebBaseLoader, AsyncChromiumLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.document_transformers import BeautifulSoupTransformer
import argparse
from pprint import pprint
from link_extractor import crawl
import time
from tqdm import tqdm
import playwright
import asyncio
import os
import json
from langsmith import traceable
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def get_args():
    """
        Parse and return the command-line arguments for the script. It includes an 
        optional argument `--use_crawling` which, if provided, will trigger the 
        crawling of the entire website. 

        Returns:
            argparse (string): the argparsed string.
    """
    parser = argparse.ArgumentParser(description='Langchain documents for webpages in the list')
    parser.add_argument("--use_crawling", action='store_true',
                        help="Runs again crawling for whole website. It might take a few minutes.")
    
    return parser.parse_args()


def load_json():
    """
        Load and return data from the 'all_valid_urls.json' file.

        Reads the JSON file containing a list of URLs and returns the data.

        Returns:
            loaded_list (list): A list of URLs loaded from the JSON file.
    """

    print("LOADING PAGES...")
    with open("all_valid_urls.json", "r") as file:
        loaded_list = json.load(file)

    return loaded_list


def async_html_loader(all_links):
    """
        Load and transform HTML content from a list of URLs asynchronously.
        Uses AsyncHtmlLoader to load HTML content and then transforms it into text format.

        Args:
            all_links (list): A list of URLs to load and transform.

        Returns:
            docs_transformed: A list of LangChain-like Documents transformed from HTML to human-readable-like text.
    """
    loader = AsyncHtmlLoader(all_links)
    docs = loader.load()

    html2text = Html2TextTransformer(ignore_links=False)
    docs_transformed = html2text.transform_documents(docs)

    return docs_transformed


def get_products(url):
    """
        Scrape product details from a given URL. It loads the webpage and extract product information 
        such as name, price, description, availability and the original price if the product contains this
        information.

        Args:
            url (str): The URL of the product page to scrape.

        Returns:
            dict: A dictionary containing the product's details (name, price, description, 
                availability, and original price if available).
    """
        
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.content()

        soup = BeautifulSoup(content, 'html.parser')

        product_container = soup.find('div', class_='pdp-main')

        product_dict = {}

        if product_container:
            product_dict['product_name'] = product_container.find('h1', class_='product-name').text.replace("\n", "")
            product_dict['price'] = product_container.find('span', class_='price-sales').text.replace("\n", "")
            product_dict['description'] = product_container.find('div', class_='product-desc').text
            product_dict['availability'] = product_container.find('div', class_='availability-msg').text.replace("\n", "")

            discount_element = product_container.find('s', class_='price-standard')
            if discount_element:
                product_dict['original_price'] = discount_element.text
    
    return product_dict


def get_table_data(url):
    """
        Extracts table data from a webpage URL. It loads the webpage, find all HTML tables, and 
        extract their contents into a structured dictionary format.

        Args:
            url (str): The URL of the webpage containing tables to scrape.

        Returns:
            all_tables_data (dict or None): A dictionary with each table's data indexed 
            by 'Table_1', 'Table_2', etc. If no tables are found, returns None.
    """

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.content()

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
    

def get_breadcrumbs(url):
    """
        Extract breadcrumb information from a product page URL.

        It load the webpage and retrieve breadcrumb elements that represent the product's type and class.

        Args:
            url (str): The URL of the product page to scrape.

        Returns:
            breadcrumbs_metadata (dict): A dictionary containing breadcrumb metadata with keys 
            'product_type' and 'product_class' if available.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.content()

        soup = BeautifulSoup(content, 'html.parser')
        raw_breadcrumbs = soup.find_all('a', class_='breadcrumb-element')
        breadcrumbs = [breadcrumb.text for breadcrumb in raw_breadcrumbs]

        breadcrumbs_metadata = {}

        if len(breadcrumbs) == 2:
            breadcrumbs_metadata["product_type"] = breadcrumbs[1]
        elif 2 < len(breadcrumbs) <= 3:
            breadcrumbs_metadata["product_type"] = breadcrumbs[1]
            breadcrumbs_metadata["product_class"] = breadcrumbs[2]

    return breadcrumbs_metadata

# def html_loader_product_data(all_links):
#     docs = []

#     for link in tqdm(all_links):
#         loader = AsyncHtmlLoader(link)
#         doc = loader.load()
#         try:
#             doc[0].metadata.update(get_breadcrumbs(link))
#             doc[0].metadata.update(get_products(link))
#         except Exception as e:
#             print(f"NOT CONSIDERED because {e}")

#         docs += doc

#     html2text = Html2TextTransformer(ignore_links=False)
#     docs_transformed = html2text.transform_documents(docs)

#     return docs_transformed

def html_loader_product_data(all_links):
    """
        Loads HTML content from a list of URLs and extracts product data and breadcrumbs.

        This function uses an asynchronous HTML loader to fetch data from the provided URLs. It updates 
        each document's metadata with product information and breadcrumbs, then transforms the documents 
        into plain text format.

        Args:
            all_links (list): A list of URLs to load and process.

        Returns:
            docs_transformed (list): A list of documents with extracted and transformed product data.
    """

    loader = AsyncHtmlLoader(all_links)
    docs = loader.load()

    for doc in tqdm(docs):
        try:
            doc.metadata.update(get_breadcrumbs(doc.metadata['source']))
            doc.metadata.update(get_products(doc.metadata['source']))
        except Exception as e:
            print(f"NOT CONSIDERED because {e}")


    html2text = Html2TextTransformer(ignore_links=False)
    docs_transformed = html2text.transform_documents(docs)

    return docs_transformed


if __name__ == '__main__':

    # BREADCRUMBS AND PRODUCT FOR METADATA TEST   

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
    docs_transformed = html_loader_product_data(all_links)
    all_tables = get_table_data(contact_url)
    if all_tables:
        tables_doc = Document(page_content=str(all_tables), metadata={"source": contact_url})
        docs_transformed.append(tables_doc)
    final_time = time.time() - start_time

    print(f"Time in loading docs: {final_time}")

    with open("metadatas2.txt", 'w') as f:
        for doc in docs_transformed:
            f.write(str(doc.metadata))
            f.write("\n")