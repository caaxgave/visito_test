import requests
from bs4 import BeautifulSoup

from urllib.parse import urljoin, urlparse
import time
from tqdm import tqdm
import json
import asyncio
from aiohttp import ClientSession


async def fetch(session, url):
    """
        Asynchronously fetches the HTML content of a URL.

        This function makes an HTTP GET request to the specified URL using an asynchronous session.
        It handles common HTTP errors and returns the response text if successful.

        Args:
            session (object): The active session for making HTTP requests.
            url (str): The URL to fetch.

        Returns:
            str or None: The HTML content of the page if the request is successful, otherwise None.
    """
    try:
        async with session.get(url) as response:
            if response.status == 404:
                return None
            elif response.status != 200:
                return None
            return await response.text()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def is_valid(url):
    """Check if the URL is a valid link to be crawled."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


async def get_all_website_links(session, url):
    """
        Asynchronously retrieves all valid internal links from a given webpage.

        This function fetches the HTML content of the specified URL, parses it to extract
        all anchor tags, and collects valid, unique internal links belonging to the same domain.

        Args:
            session (object): The active session for making HTTP requests.
            url (str): The URL of the webpage to scrape for links.

        Returns:
            set: A set of valid, unique internal links found on the webpage.
    """
    urls = set()
    domain_name = urlparse(url).netloc

    html_content = await fetch(session, url)
    if html_content is None:
        return urls

    soup = BeautifulSoup(html_content, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")

        if href == "" or href is None:
            continue
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path

        if not is_valid(href):
            continue
        if domain_name not in href:
            continue
        urls.add(href)
    return urls


async def crawl(url, max_urls=400):
    """
        Asynchronously crawls the website, collecting valid URLs up to a specified limit.

        This function starts from a given URL and traverses the website to gather internal links,
        ignoring media files. It continues until the maximum number of URLs is reached or no more
        links are found.

        Args:
            url (str): The starting URL for the crawl.
            max_urls (int, optional): The maximum number of URLs to crawl. Defaults to 400.

        Returns:
            list: A list of valid URLs discovered during the crawl.
    """

    print("CRAWLING...")
    visited_urls = set()
    urls_to_visit = set([url])
    valid_urls = []
    media_files = (".jpg", ".png", ".mov", ".mp3", ".gif", ".tiff")
    
    async with ClientSession() as session:
        with tqdm(total=max_urls, desc="Crawling Progress", unit="url") as pbar:
            while urls_to_visit and len(visited_urls) < max_urls:
                current_url = urls_to_visit.pop()
                if (current_url in visited_urls) or (current_url.endswith(media_files)):
                    #print("URL ignored: ", current_url)
                    continue

                #print(f"Crawling: {current_url}")
                visited_urls.add(current_url)
                links = await get_all_website_links(session, current_url)

                if links != set():
                    valid_urls.append(current_url)

                urls_to_visit.update(links - visited_urls)
                pbar.update(1)
                await asyncio.sleep(0.5)

    return valid_urls


def save_as_json(valid_urls):
    """
        Saves a list of valid URLs to a JSON file.

        Args:
            valid_urls (list): The list of URLs to save.

    """
    with open("all_valid_urls.json", "w") as file:
        json.dump(valid_urls, file)


if __name__ == '__main__':
    url = "https://www.conairmexico.com/"

    start_time = time.time()
    all_links = asyncio.run(crawl(url))
    total_time = time.time() - start_time
    print("SAVING AS JSON...")
    save_as_json(all_links)
    print("Total links: ", len(all_links))
    print("Time on crawling: " ,total_time)

    
