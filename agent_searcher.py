import os
import ast
import requests
from serpapi import GoogleSearch
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from bs4 import BeautifulSoup
from langchain.utilities import GoogleSearchAPIWrapper, SerpAPIWrapper

from langchain.tools.google_search.tool import GoogleSearchResults, GoogleSearchRun
from langchain.agents import load_tools, initialize_agent, Tool, ZeroShotAgent, AgentExecutor
from langchain import OpenAI, LLMChain
import concurrent.futures
from fake_useragent import UserAgent
import http.client
from scrapingbee import ScrapingBeeClient
import time
http.client._MAXHEADERS = 1000
import json
import re
import numpy as np

from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
#Helper methods
def split_text_into_chunks(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

def remove_urls(text):
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.sub(r'', text)


def remove_emails(text):
    email_pattern = re.compile(r'\S+@\S+\.\S+')
    return email_pattern.sub(r'', text)


def remove_phone_numbers(text):
    phone_pattern = re.compile(r'\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
    return phone_pattern.sub(r'', text)


def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)


def get_coordinates(query):
    pattern = r"(-?\d+\.\d+),\s*(-?\d+\.\d+)"

    match = re.search(pattern, query)

    if match:
        coordinates = [float(match.group(1)), float(match.group(2))]
    else:
        coordinates = [29.518000813829737, -98.59083760994389]
    return coordinates
class WebSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Web Search",
            description=("Useful when you need to search the web for more information about a place"),
        )

    async def _arun(self, query, num_results=5):
        pass

    def _run(self, query):
        urls = self.get_search_results(query)
        extracted_info = self.extract_information(urls, query)
        return extracted_info

    def get_link(self, r):
        return r['link']

    def get_geocode_data(lat, lng):
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&result_type=locality&key={api_key}"
        response = requests.get(url)
        data = response.json()
        locality = data['results'][0]['formatted_address']
        return locality
    
    def get_search_results(self, query):
        coordinates  = get_coordinates(query)

        search = GoogleSearch({
            "q": query, 
            "logging": False
        })

        result = search.get_dict()
        return list(map(self.get_link, result['organic_results']))
    
    def extract_information(self, search_results, query):
        extracted_info = []
        
        template = """You are an AI language model trained to find information within text. 
        Given the following text content extracted from a web page, please locate and provide all the 
        relevant pieces of information related to the query: 
        Web page text content: {chunk}
        query: '{input}'
        """
        prompt = PromptTemplate(
            input_variables=["input", "chunk"],
            template=template,
        )

        llm_chain=LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
        small_test = search_results[:2]
        print("Urls: ",small_test)
        for url in small_test:
            print("Get page content for this url: ",url)
            response = client.get(url)
        
            extracted_info = []
            if response.status_code == 200:
                print("Process data for this url: ",url)
                page_content = response.text
                soup = BeautifulSoup(page_content, 'html.parser')
                page_text = soup.get_text()
                page_text = remove_urls(page_text)
                page_text = remove_emails(page_text)
                page_text = remove_phone_numbers(page_text)
                page_text = remove_special_characters(page_text)
                page_text = page_text.replace("\n", " ")
                page_text = page_text.replace("\t", " ")
                page_text = " ".join(page_text.split())
                
                if len(page_text) > 15000:
                    return extracted_info
                cleaned_text = " ".join(page_text.split())
                extracted_data = self.process_with_llm(cleaned_text, llm_chain, query)
                extracted_info.append(extracted_data)

        return extracted_info

    def make_llm_prediction(self, chunk, llm_chain, query):
        print("Make LLM Prediction on a chunk")
        llm_input = {
            "input": query,
            "chunk": chunk,
        }
        response = llm_chain.predict(**llm_input)
        print("Response: ",response)
        result = response.split("Final Answer")[-1].strip()
        return result

    def process_with_llm(self, page_content, llm_chain, query):
        print("Process a page results")
        chunk_size = 1250
        chunks = split_text_into_chunks(page_content, chunk_size)
        results = []
        for chunk in chunks:
            result = self.make_llm_prediction(chunk, llm_chain, query)
            results.append(result)

        combined_response = '\n'.join(results)
        return combined_response


def extract_new_sentences(arr):
    new_sentences = []
    for item in arr:
        new_sentence = item.split(': ')[1]
        new_sentences.append(new_sentence)
    return new_sentences

def calculate_similarity_score(query_result, query_result1):
    sim_score = np.dot(query_result, query_result1)
    return sim_score

def filter_similar_texts(text_list, threshold=0.95):
    
    embeddings_list = [embeddings.embed_query(text) for text in text_list]
    
    to_remove = set()
    for i in range(len(text_list)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(text_list)):
            sim_score = calculate_similarity_score(embeddings_list[i], embeddings_list[j])
            if sim_score > threshold:
                to_remove.add(j)

    filtered_list = [text for index, text in enumerate(text_list) if index not in to_remove]
    return filtered_list

def get_subqueries(query):
    llm=OpenAI(temperature=0.0, model_name='gpt-3.5-turbo')

    prompt = f"For each noun in input return \nAdjective+Noun \nor else Noun \nInput: {query}"
    
    val = llm.generate([prompt])
    try:
        response = val.generations[0][0].text
        query_variations = response.split('\n')
        # query_variations = extract_new_sentences(response.split('\n'))
        if len(query_variations) != 0:
            new_list = filter_similar_texts(query_variations)
            return new_list
        else:
            return query
    except:
        return query

def search_places(query):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    coordinates = get_coordinates(query)
    keyword = query.split("|")[0]
    params = {
        "keyword": keyword,
        "location": f"{coordinates[0]},{coordinates[1]}",
        "radius": 1000,
        "key": api_key,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        places = [(place["name"],place['geometry']['location']) for place in data["results"]]
        place_names = [place[0] for place in places]
        return "|".join(place_names)
    else:
        raise Exception(f"Request failed with status code {response.status_code}")



tools = [
    Tool(
        name="SearchPlaces",
        func = search_places,
        description="Useful when you need to search maps or places API, output is a list of places",
    ),
    WebSearchTool()
]


s = time.perf_counter()
questions = get_subqueries("Kid friendly restaurants near a park")
new_questions = []
for question in questions:
    new_questions.append(question + f", coordinates: 33.646451, -117.639262")


def generate_serially(qs):
    for q in qs:
        agent = initialize_agent(tools, OpenAI(temperature=0,model_name="gpt-3.5-turbo"), agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)
        response = agent({"input":q})
        print(response)

generate_serially(new_questions)