import streamlit as st
from dataclasses import dataclass
from py2neo import Graph
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import re
import yaml

# Load configuration
def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_config()

# Neo4j connection
auth = (config["neo4j"]["user"], config["neo4j"]["password"])
graph = Graph(config["neo4j"]["uri"], auth=auth)

# Embeddings and LLM
embeddings = OllamaEmbeddings(model='llama3:8b')
llm = Ollama(model="llama3:8b")
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Extract symptoms from query
def extract_symptoms_from_query(query):
    pattern = r'WHERE s\.name IN \[(.*?)\]'
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        symptoms_str = match.group(1)
        symptoms = [symptom.strip().strip('"').strip("'") for symptom in symptoms_str.split(',')]
        return symptoms
    return None

# Read query from Neo4j
def read_query(query):  
    try:
        result = graph.run(query)
    except Exception as e:
        result = str(e)
    return result

# Generate Cypher query from prompt
def generate_cypher_query(prompt):
    system_prompt = f"""
    You are an experienced graph databases developer.
    This is the schema representation of a Neo4j database.
    Provide answers in Cypher based on the following schema and example.
    ### Important: when ever you use node Drug denote its object with d eg.(d:Drug)
                   -Disease node's object is g eg. (g:Disease)
                   -Symptom node's object is s eg. (s:Symptom)
                   -Composition node's object is c eg. (c:Composition)
    ### Give only one cypher text.
    ### The Schema
    {schema}
    ### ONLY PROVIDE THE CYPHER TEXT NO EXPLANATION NOTHING
    ### Example
    Question:
    ->find alternate of zometa drug
    cypher text- MATCH p=(d:Drug)-[:HAS_COMPOSITION]->(c:Composition)<-[:HAS_COMPOSITION]-(d2:Drug) where d.name='zometa' return d2.name;
    -> Give the drugs for cancer disease.
    cypher text- MATCH p=(d:Drug)-[:TREATS]->(g:Disease) where g.name='cancer' return d.name Limit 5;
    -> I have chest pain, vomiting, nausea. what diseas may i have?
    cypher text- MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptoms) where s.name IN ["vomiting","increased thirst","nausea"] return DISTINCT d.name;
    ### Give answer to the following quetions based on above examples.
    {prompt}
    """
    response = llm.invoke(system_prompt)
    return response

# Process response
def process_response(response):
    if 's.name IN' in response:
        symptoms = extract_symptoms_from_query(response)
        updated_symptoms = []
        for query in symptoms:
            docs = db3.similarity_search(query)
            updated_symptoms.append(docs[0].page_content)
        symptom_list = '["' + '", "'.join(symptoms) + '"]'
        updated_response = response.replace(symptom_list, f'{updated_symptoms}')
        ans = read_query(updated_response)
    else:
        ans = read_query(response)
    return ans

# Generate natural language from results
def generate_natural_language_from_results(results):
    formatted_results = "\n".join([str(result) for result in results])
    nl_prompt = f"""
    You are an AI language model specifically designed for recommendation.
    Convert the following Cypher query results into a natural language description.
    Description should be a brief paragraph and patient friendly as a doctor is recommending something to a patient.

    Results:
    {formatted_results}

    Description:
    """
    nl_response = llm.invoke(nl_prompt)
    return nl_response

# Streamlit app starts here
def main():
    st.title("Drug recommendation system")

    api_key = st.text_input("Insert API Key")

    if api_key:
        start_time = st.session_state.get('start_time')
        if start_time is None:
            bot = None
            st.session_state['main_obj'] = bot

        if MESSAGES not in st.session_state:
            st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi!")]

        msg: Message
        for msg in st.session_state[MESSAGES]:
            st.chat_message(msg.actor).write(msg.payload)

        prompt: str = st.chat_input("Enter here")
        if prompt:
            st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
            st.chat_message(USER).write(prompt)
            with st.spinner("Typing......"):
                cypher_query = generate_cypher_query(prompt)
                results = process_response(cypher_query)
                if results:
                    ans_list = results.data()
                    if ans_list:
                        response = generate_natural_language_from_results(ans_list)
                    else:
                        response = "No results found."
                else:
                    response = "No results found."
                st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
                st.chat_message(ASSISTANT).write(response)

if __name__ == "__main__":
    main()
