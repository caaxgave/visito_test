import requests
from bs4 import BeautifulSoup

from urllib.parse import urljoin, urlparse
import time
from tqdm import tqdm
import json
import asyncio
from aiohttp import ClientSession


async def fetch(session, url):
    """Fetch the content of a URL asynchronously and handle 404s."""
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
    """Returns all URLs that is found on `url` in which it belongs to the same website."""
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
    """Asynchronous crawler that collects all internal URLs from a website."""

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
    with open("all_valid_urls.json", "w") as file:
        json.dump(valid_urls, file)


if __name__ == '__main__':
    url = "https://www.conairmexico.com/"

    start_time = time.time()
    print("CRAWLING...")
    all_links = asyncio.run(crawl(url))
    total_time = time.time() - start_time
    print("SAVING AS JSON...")
    save_as_json(all_links)
    print("Total links: ", len(all_links))
    print("Time on crawling: " ,total_time)

    
