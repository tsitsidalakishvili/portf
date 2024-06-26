import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from jira import JIRA
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import subprocess
import plotly.graph_objects as go
import io
import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import os
import base64
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
import constants
from neo4j import GraphDatabase



# Replace 'your_api_key_here' with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'

# Initialize session state for storing fetched data
if 'fetched_data' not in st.session_state:
    st.session_state['fetched_data'] = []


# Initialize Neo4j graph
graph = Neo4jGraph(
    url="bolt://44.200.10.235:7687",  # Neo4j database URL
    username="neo4j",  # Username for Neo4j
    password="housefalls-coxswain-ones"  # Password for Neo4j
)

# Initialize the Language Model and Chain
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),  # Adjust temperature as needed
    graph=graph,
    verbose=True
)



# Streamlit App Layout
st.title('Connect Data To LLM')

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = {
        'jira_df': pd.DataFrame(),
        'prompt_templates': [],
        'openai_api_key': '',
        'selected_columns': []
    }
    

def initialize_default_prompt_templates():
    if 'prompt_templates' not in st.session_state.data:
        st.session_state.data['prompt_templates'] = []
    templates = [
        {
            "name": "Neo4j Query",
            "instructions": "As a Neo4j user, provide a detailed report based on a specific query.",
            "example_input": "Provide a detailed report based on the query: MATCH (p:Person)-[:FRIEND]->(f:Person) RETURN p, f LIMIT 5",
            "example_output": "The query results include [Number of Nodes] nodes and [Number of Relationships] relationships. The first 5 records are displayed below:",
            "query_template": "Provide a detailed report based on the query: {neo4j_query}.",
            "few_shot_count": 3
        },
        {
            "name": "Issue Tracker",
            "instructions": "As a Project Coordinator, provide an integrated report on the status, recent developments, and any blockers or dependencies related to a specific issue.",
            "example_input": "Provide a detailed report on issue KEY-123.",
            "example_output": "Issue KEY-123, titled [Summary], is currently [Status]. The latest update was on [Updated], with [Comment count] comments, and the last comment: [Latest Comment]. There are [Blocker count] blockers impeding progress, with the main dependency on [Dependency Key].",
            "query_template": "Provide a detailed report on issue {issue_key}.",
            "few_shot_count": 3
        },

        {
            "name": "Analyze",
            "instructions": "Analyze selected columns from Jira text data to identify patterns, similarities, and inefficiencies. Provide insights into repetitive tasks and potential areas for process optimization.",
            "example_input": "Analyze the impact and frequency of issues related to 'Network Connectivity' over the last quarter.",
            "example_output": "The analysis reveals that 'Network Connectivity' issues have increased by 25% over the last quarter, with most incidents reported during peak usage hours, suggesting a need for infrastructure scaling.",
            "query_template": "Analyze {issue_topic} issues over the last {time_period}.",
            "few_shot_count": 3
        },
        # New Template: Resource Allocation
        {
            "name": "Resource Allocation",
            "instructions": "As a Resource Manager, outline the workload distribution across the team, flagging any over-allocations.",
            "example_input": "How is the workload spread for the current sprint?",
            "example_output": "In the current sprint, [Assignee 1] is at 80% capacity, [Assignee 2] is over-allocated by 20%, and [Assignee 3] can take on more work. Adjustments are recommended.",
            "query_template": "How is the workload spread for the current sprint?",
            "few_shot_count": 3
        },
        # New Template: Risk Assessment
        {
            "name": "Risk Assessment",
            "instructions": "As a Risk Assessor, identify high-risk issues based on priority, due dates, and current status.",
            "example_input": "What are the high-risk issues for the upcoming release?",
            "example_output": "High-risk issues for the release include [Issue 1] due in [Days] days at [Priority] priority, and [Issue 2] which is [Status] and past the due date.",
            "query_template": "What are the high-risk issues for the upcoming release?",
            "few_shot_count": 3
        },
    ]
    if not st.session_state.data['prompt_templates']:
        st.session_state.data['prompt_templates'] = templates
        
initialize_default_prompt_templates()


# JIRA Utility Functions
@st.cache(suppress_st_warning=False, allow_output_mutation=True)
def fetch_project_names(username, api_token, jira_domain):
    base_url = f'{jira_domain}/rest/api/latest/project'  # Use jira_domain in the URL
    headers = {'Accept': 'application/json'}
    try:
        response = requests.get(base_url, auth=(username, api_token), headers=headers)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        data = response.json()
        return [project['name'] for project in data]
    except requests.exceptions.RequestException as e:
        st.error(f'Failed to fetch project names: {e}')
        return []

@st.cache(suppress_st_warning=False, allow_output_mutation=True)
def fetch_jira_issues(username, api_token, project_name, jql_query, jira_domain):
    base_url = f'{jira_domain}/rest/api/latest/search'  # Use jira_domain in the URL
    params = {
        'jql': jql_query,
        'maxResults': 1000,
        'fields': '*all'
    }
    try:
        response = requests.get(base_url, auth=(username, api_token), headers={'Accept': 'application/json'}, params=params)
        response.raise_for_status()
        data = response.json()
        return pd.json_normalize(data['issues'])
    except requests.exceptions.RequestException as e:
        st.error(f'Failed to fetch issues: {e}')
        return pd.DataFrame()

# Add/Edit Prompt Template Functions
def add_template(name, instructions, example_input, example_output, query_template, few_shot_count):
    template = {
        "name": name,
        "instructions": instructions,
        "example_input": example_input,
        "example_output": example_output,
        "query_template": query_template,
        "few_shot_count": few_shot_count
    }
    st.session_state.data['prompt_templates'].append(template)

def update_template(index, name, instructions, example_input, example_output, query_template, few_shot_count):
    st.session_state.data['prompt_templates'][index] = {
        "name": name,
        "instructions": instructions,
        "example_input": example_input,
        "example_output": example_output,
        "query_template": query_template,
        "few_shot_count": few_shot_count
    }

# Construct Full Prompt with Dynamic Few-Shot Examples
def construct_full_prompt(template, actual_input):
    # Incorporate few-shot examples dynamically based on the template's 'few_shot_count'
    few_shot_examples = "\n".join([f"Example Input: {template['example_input']}\nExample Output: {template['example_output']}" for _ in range(template['few_shot_count'])])
    return f"{template['instructions']}\n{few_shot_examples}\n{template['query_template']}\n\n{actual_input}"



def execute_prompt(template, test_input, data, selected_columns):
    try:
        # Filter the data to include only the selected columns
        if selected_columns:
            data_df = pd.read_json(data)
            filtered_data = data_df[selected_columns].to_json()
        else:
            filtered_data = data

        full_prompt = f"{template}\n\n{test_input}\n\n{filtered_data}"

        # Split the full_prompt into segments of appropriate length
        segments = [full_prompt[i:i+4096] for i in range(0, len(full_prompt), 4096)]

        responses = []

        for segment in segments:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": segment}]
            )
            responses.append(response.choices[0].message['content'])

        return " ".join(responses)
    except Exception as e:
        return f"Error: {e}"
    


def execute_neo4j_prompt(chain, query_template, query_parameters):
    """
    Executes a Cypher query using GraphCypherQAChain and generates a response.
    
    Args:
    - chain (GraphCypherQAChain): The initialized GraphCypherQAChain for Neo4j.
    - query_template (str): The Cypher query template.
    - query_parameters (dict): Parameters to format the Cypher query template.
    
    Returns:
    - str: The generated response based on the Cypher query results.
    """
    # Format the Cypher query using the provided parameters
    cypher_query = query_template.format(**query_parameters)
    
    # Execute the query using the chain
    response = chain.run(cypher_query)
    
    return response


# Neo4j Connection Class
class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(user, pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, parameters=None, db=None):
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response


# Function to get database session
def get_db_session(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver.session()



# Fetch Data from Neo4j Function
def fetch_data_from_neo4j(uri, user, pwd, query):
    # Initialize the Neo4j connection with user-entered credentials
    neo4j_conn = Neo4jConnection(uri, user, pwd)
    
    # Execute the query
    result = neo4j_conn.query(query)
    
    # Convert the result to a pandas DataFrame or any desired format
    data = pd.DataFrame([dict(record) for record in result])
    
    # Close the Neo4j connection
    neo4j_conn.close()
    
    return data




# Sidebar for User Input and Mode Selection
with st.sidebar:
    st.title("Settings")
    app_mode = st.selectbox('Choose the Mode', ['How it works', 'Connect to your Data Sources', 'Create DB', 'Manage Prompts'])




# ---------------------------------------------------------------------------------------------------------------------------------# 

# Data Source Management
if app_mode == 'Connect to your Data Sources':
    with st.sidebar:
        data_source = st.radio("Select Data Source", ['Upload CSV', 'Jira', 'Neo4j', 'Connect to API'])
    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
        if uploaded_file is not None:
            try:
                st.session_state.data['jira_df'] = pd.read_csv(uploaded_file)
                st.success('CSV file uploaded successfully!')
                st.write(st.session_state.data['jira_df'])
            except Exception as e:
                st.error(f'Failed to read CSV file: {e}')
        else:
            st.info("Upload a CSV file to proceed.")



    if data_source == 'Jira':
        with st.sidebar:
            jira_domain = st.text_input('Jira Domain', 'https://your-jira-domain.atlassian.net')
            username = st.text_input('Jira username')
            api_token = st.text_input('Jira API token', type='password')
        if username and api_token:
            project_names = fetch_project_names(username, api_token, jira_domain)
            if project_names:
                with st.sidebar:
                    selected_project_name = st.selectbox('Select a Jira project', project_names)
                jql_query = st.text_area("Enter JQL Query:")
                if st.button('Fetch Jira Issues'):
                    with st.spinner('Fetching Jira Issues...'):
                        st.session_state.data['jira_df'] = fetch_jira_issues(username, api_token, selected_project_name, jql_query, jira_domain)
                        if not st.session_state.data['jira_df'].empty:
                            st.success('Jira issues fetched successfully!')
                            # Display Raw Data (Option: As JSON)
                            st.json(st.session_state.data['jira_df'].to_json(orient="records"))
                        else:
                            st.error('No data available for the selected project.')
            else:
                st.error("Check your credentials.")

        if not st.session_state.data['jira_df'].empty:
            st.write("Fetched Jira Data")
            st.dataframe(st.session_state.data['jira_df'])



    if data_source == 'Neo4j':
        # User inputs for Neo4j credentials and query
        uri = st.sidebar.text_input("Neo4j URI", "bolt://3.237.232.235:7687")
        user = st.sidebar.text_input("Neo4j Username", "neo4j")
        pwd = st.sidebar.text_input("Neo4j Password", type="password")
        neo4j_query = st.text_area("Enter Neo4j Query:")
        
        # Button to fetch data using entered credentials and query
        if st.button('Fetch Data from Neo4j'):
            with st.spinner('Fetching data from Neo4j...'):
                neo4j_data = fetch_data_from_neo4j(uri, user, pwd, neo4j_query)
                st.session_state.data['neo4j_df'] = neo4j_data
                st.success('Data fetched from Neo4j successfully!')
                st.write(neo4j_data)


    if data_source == 'Connect to API':
        # Define input fields for query parameters
        key = st.text_input("Key (Name/Keyword)")
        firstname = st.text_input("First Name")
        lastname = st.text_input("Last Name")
        organization_ids = st.multiselect("Organization IDs", [22, 23])
        position_ids = st.multiselect("Position IDs", [22, 23])
        year_selected_values = st.multiselect("Year Selected Values", [2020, 2021, 2022, 2023])

        if st.button("Fetch Data"):
            base_url = "https://declaration.acb.gov.ge/Api/Declarations"
            params = {
                "Key": key,
                "Firstname": firstname,
                "Lastname": lastname,
                "OrganizationIds": organization_ids,
                "PositionIds": position_ids,
                "YearSelectedValues": year_selected_values
            }
            response = requests.get(base_url, params=params)

            if response.status_code == 200:
                st.session_state['fetched_data'] = response.json()
                fetched_count = len(st.session_state['fetched_data'])  # Get the count of fetched records
                if fetched_count > 0:
                    st.success(f"Fetched {fetched_count} records.")
                    for index, record in enumerate(st.session_state['fetched_data'], start=1):
                        st.write(f"Record {index}:")
                        st.write(record)
                else:
                    st.warning("No data found for the given parameters.")
            else:
                st.error(f"Error fetching data. Status code: {response.status_code}")


    if not st.session_state.data['jira_df'].empty:
        st.title("Select Columns for Analysis")
        selected_columns = st.multiselect("Select columns to save for analysis:", st.session_state.data['jira_df'].columns)
        st.session_state.data['selected_columns'] = selected_columns



# ---------------------------------------------------------------------------------------------------------------------------------# 

if app_mode == 'Create DB':
    st.title("Create/Populate Neo4j Database")
    if st.session_state['fetched_data']:
        st.subheader("Review and Confirm Data Model")
        for index, record in enumerate(st.session_state['fetched_data'], start=1):
            st.write(f"Record {index} - Person Node:")
            st.json({
                "ID": record.get('Id'),
                "First Name": record.get('FirstName'),
                "Last Name": record.get('LastName'),
                "Birth Place": record.get('BirthPlace'),
                "Birth Date": record.get('BirthDate')
            })
            if "Jobs" in record:
                for job in record["Jobs"]:
                    st.write("Works At (Relationship):")
                    st.json({
                        "Person": f"{record.get('FirstName')} {record.get('LastName')}",
                        "Organization": job.get('Organisation'),
                        "Position": job.get('Position')
                    })
            if "Properties" in record:
                for prop in record["Properties"]:
                    st.write("Owns Property (Relationship):")
                    st.json({
                        "Person": f"{record.get('FirstName')} {record.get('LastName')}",
                        "Address": prop.get('Address'),
                        "Area": prop.get('Area'),
                        "Property Type": prop.get('PropertyType')
                    })

        uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
        user = st.text_input("User", "neo4j")
        password = st.text_input("Password", type="password")

        if st.button("Populate Database"):
            try:
                session = get_db_session(uri, user, password)
                for record in st.session_state['fetched_data']:
                    person_id = record.get("Id")
                    session.run("""
                    MERGE (p:Person {id: $id})
                    ON CREATE SET p.firstName = $firstName, p.lastName = $lastName,
                                  p.birthPlace = $birthPlace, p.birthDate = $birthDate
                    """, {
                        "id": person_id,
                        "firstName": record["FirstName"],
                        "lastName": record["LastName"],
                        "birthPlace": record["BirthPlace"],
                        "birthDate": record["BirthDate"]
                    })
                    for job in record.get("Jobs", []):
                        org_name = job["Organisation"]
                        session.run("""
                        MERGE (o:Organization {name: $orgName})
                        WITH o
                        MATCH (p:Person {id: $id})
                        MERGE (p)-[:WORKS_AT]->(o)
                        """, {
                            "id": person_id,
                            "orgName": org_name
                        })
                    for prop in record.get("Properties", []):
                        address = prop["Address"]
                        session.run("""
                        MERGE (pr:Property {address: $address})
                        ON CREATE SET pr.area = $area, pr.propertyType = $propertyType,
                                      pr.purchaseForm = $purchaseForm, pr.purchaseDate = $purchaseDate,
                                      pr.currencyName = $currencyName
                        WITH pr
                        MATCH (p:Person {id: $id})
                        MERGE (p)-[:OWNS_PROPERTY {price: $price}]->(pr)
                        """, {
                            "id": person_id,
                            "address": address,
                            "area": prop["Area"],
                            "propertyType": prop["PropertyType"],
                            "purchaseForm": prop.get("PurchaseForm", "Undefined"),
                            "purchaseDate": prop.get("PurchaseDate", "Undefined"),
                            "price": prop.get("Price", 0),  # Defaulting to 0 if Price not available
                            "currencyName": prop.get("CurrencyName", "Undefined")
                        })
                st.success("Database populated successfully!")
            except Exception as e:
                st.error(f"Failed to populate the database: {e}")
            finally:
                if 'session' in locals():
                    session.close()
    else:
        st.warning("No data available. Please fetch data first.")




# ---------------------------------------------------------------------------------------------------------------------------------# 



if app_mode == 'Manage Prompts':
    with st.sidebar:
        openai_api_key = st.text_input('OpenAI API key', type='password')
        if openai_api_key:
            st.session_state.data['openai_api_key'] = openai_api_key
            openai.api_key = openai_api_key

    # Select Data Source
    data_sources = ['Jira', 'Neo4j', 'CSV', 'API']
    st.session_state['data_source'] = st.selectbox('Select Data Source', data_sources)

    # Display only the selected data source's DataFrame or data
    if st.session_state['data_source'] == 'Jira':
        if 'jira_df' in st.session_state and not st.session_state['jira_df'].empty:
            st.write("Jira Data:")
            st.dataframe(st.session_state['jira_df'])
        else:
            st.warning("No Jira data available. Please connect to Jira data source first.")
    elif st.session_state['data_source'] == 'Neo4j':
        if 'neo4j_df' in st.session_state and not st.session_state['neo4j_df'].empty:
            st.write("Neo4j Data:")
            st.dataframe(st.session_state['neo4j_df'])
        else:
            st.warning("No Neo4j data available. Please connect to Neo4j data source first.")
    elif st.session_state['data_source'] == 'CSV':
        if 'csv_df' in st.session_state and not st.session_state['csv_df'].empty:
            st.write("CSV Data:")
            st.dataframe(st.session_state['csv_df'])
        else:
            st.warning("No CSV data available. Please upload a CSV file first.")
    elif st.session_state['data_source'] == 'API':
        if 'api_data' in st.session_state and st.session_state['api_data']:
            st.write("API Data:")
            st.json(st.session_state['api_data'])
        else:
            st.warning("No API data available. Please fetch data from an API first.")

 

    if data_source == 'Neo4j' and app_mode == 'Manage Prompts':
        # Example Cypher query template and parameters
        query_template = "MATCH (p:Person)-[:FRIEND]->(f:Person) RETURN p, f LIMIT {limit}"
        query_parameters = {"limit": 5}  # Example parameter, adjust as needed based on user input
        
        if st.button('Execute Neo4j Prompt'):
            response = execute_neo4j_prompt(chain, query_template, query_parameters)
            st.write(response)



    # Display only the selected columns from the saved DataFrame
    if 'selected_columns' in st.session_state.data and st.session_state.data['selected_columns']:
        st.write("Selected Columns for Analysis:")
        st.write(st.session_state.data['selected_columns'])
        
        if not st.session_state.data['jira_df'].empty or not st.session_state.data['neo4j_df'].empty:
            st.write("Saved DataFrame (Selected Columns):")
            # Determine which DataFrame to use based on what's available
            data_source_df = st.session_state.data['jira_df'] if not st.session_state.data['jira_df'].empty else st.session_state.data['neo4j_df']
            # Filter the DataFrame to only include the selected columns
            filtered_df = data_source_df[st.session_state.data['selected_columns']]
            st.dataframe(filtered_df)
        else:
            st.write("No DataFrame saved.")
    else:
        st.write("No columns selected or DataFrame saved.")

    # Edit existing prompt template
    existing_prompt_names = [tpl['name'] for tpl in st.session_state.data['prompt_templates']]
    selected_template_idx = st.selectbox(
        "Prompt templates:", 
        range(len(existing_prompt_names)), 
        format_func=lambda x: existing_prompt_names[x]
    )
    selected_template = st.session_state.data['prompt_templates'][selected_template_idx]


    # Execute prompt with user input
    user_input = st.text_input("Enter your query or question:")

    if user_input:  # Check if user has entered any input
        if st.button("Execute Prompt"):
            selected_template = st.session_state.data['prompt_templates'][selected_template_idx]  # Moved inside the button condition
            if selected_template:
                if st.session_state.data['jira_df'].empty:
                    st.warning("No Jira data available. Please fetch Jira issues first.")
                else:
                    with st.spinner('Executing Prompt...'):  # Move the spinner inside the button condition
                        # Construct the full prompt
                        full_prompt = construct_full_prompt(selected_template, user_input)

                        # Execute the prompt with the Jira data
                        response = execute_prompt(selected_template, user_input, st.session_state.data['jira_df'].to_json(), st.session_state.data['selected_columns'])

                        # Display the response
                        st.write("Response:")
                        st.write(response)
    else:
        st.info("Please enter your query or question before executing the prompt.")


    # Edit functionality under expander
    with st.expander("Edit Prompt Template"):
        selected_template = st.session_state.data['prompt_templates'][selected_template_idx]
        edited_name = st.text_input("Prompt Name:", value=selected_template['name'], key=f"name_{selected_template_idx}", help="Provide a concise yet descriptive name. This will help users identify the prompt's purpose at a glance.")
        edited_instructions = st.text_area("Instructions:", value=selected_template['instructions'], key=f"instructions_{selected_template_idx}", help="""Detail the intended interaction model or role the AI should assume. For example, 'You are a helpful assistant that provides concise answers.' Be specific about the tone, style, and any constraints the AI should adhere to.""")
        edited_example_input = st.text_area("Example Input:", value=selected_template['example_input'], key=f"input_{selected_template_idx}", help="Include a representative input that the prompt is expected to handle. This should illustrate the kind of questions or commands the AI will respond to.")
        edited_example_output = st.text_area("Example Output:", value=selected_template['example_output'], key=f"output_{selected_template_idx}", help="Provide an example response that aligns with the instructions and input. Ensure it demonstrates the desired output format and content.")
        edited_query_template = st.text_area("Query Template:", value=selected_template['query_template'], key=f"template_{selected_template_idx}", help="""Craft the structure of the query that will be generated. Use placeholders for dynamic parts. For instance, '{user_query}' could be replaced with actual user input during execution.""")
        edited_few_shot_count = st.slider("Number of Few-Shot Examples", min_value=1, max_value=10, value=selected_template['few_shot_count'], key=f"few_shot_{selected_template_idx}", help="Adjust the number of few-shot examples. Few-shot learning helps the model understand the task by providing examples.")

        if st.button("Save Changes", key=f"save_{selected_template_idx}"):
            update_template(selected_template_idx, edited_name, edited_instructions, edited_example_input, edited_example_output, edited_query_template, edited_few_shot_count)
            st.success("Prompt template updated successfully!")




# ---------------------------------------------------------------------------------------------------------------------------------# 


# Home Page
if app_mode == 'How it works':
    st.markdown("""
    Streamlit Application for Data Analysis with LLM Integration

    This Streamlit application is designed to seamlessly integrate data from Jira, user-uploaded CSV files with the capabilities of Large Language Models (LLMs). 
    This facilitates data analysis and interaction through tailored prompts, making it easier to derive insights from various data sources.

    How It Works
    ------------

    **Data Source Integration:**
    - **Jira**: Effortlessly fetch projects and issues from your Jira account by specifying the project and customizing the JQL query to retrieve the relevant data.
    - **Upload CSV**: Provides the ability to upload your datasets through CSV files for quick and direct analysis, thereby enhancing flexibility in data integration.
    - **Neo4j**: Connect to Neo4j databases to fetch and utilize graph-based data for analysis.
    - **APIs**: Incorporate data from external APIs to enrich your analysis with real-time or specialized information.

    **Prompt Management:**
    - **Add/Edit Prompt Templates**: Users can define templates with specific instructions, example inputs/outputs, query templates, and a designated number of few-shot examples. These templates guide the LLM in generating nuanced and relevant responses based on your data.
    - **Execute Prompts**: Execute your crafted prompts against the integrated data, receiving responses generated by the LLM. This dynamic interaction enables the extraction of insights and information tailored to your specific queries.

    ### Tips for Effective Prompt Engineering

    - **Be Specific**: Clearly articulate the role and scope of knowledge you expect the LLM to assume in your prompts to ensure accuracy and relevance in the responses.
    - **Utilize Few-Shot Learning**: Incorporate examples into your prompts to guide the LLM. This technique, known as few-shot learning, can significantly enhance the model's understanding and the quality of its output.
    - **Dynamic Queries**: Harness the power of dynamic variables in your templates to make your prompts adaptable to a variety of queries and data points, ensuring versatility in your analysis.

    The integration of data sources with LLMs opens up new avenues for data analysis, offering a unique blend of flexibility, depth, and interactivity. Explore the capabilities of this application to connect data to LLMs for an enhanced data analysis experience, tailored to meet the evolving needs of users seeking to leverage the vast potential of large language models.


    """)