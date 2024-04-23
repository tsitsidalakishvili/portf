import streamlit as st
import requests
from neo4j import GraphDatabase
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
import os
import constants


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

page = st.sidebar.selectbox("Select a page", ("Fetch", "Create DB", "Natural Language Query"))

# Function to get database session
def get_db_session(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver.session()

if page == "Fetch":
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


if page == "Create DB":
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




if page == "Natural Language Query":
    st.title("Query with Natural Language")
    
    # Setup for Langchain and OpenAI (Assuming you have an OpenAI API key)
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key here:", type="password")
    langchain_llm = OpenAI(api_key=openai_api_key)
    
    st.write("### Ask a Question")
    st.write("Use the text area below to type your question or select from the examples provided.")

    # Example queries
    example_queries = [

        # Add more example queries as needed
    ]
    example = st.selectbox("Example Queries", [""] + example_queries)

    # Pre-populate the question field with the selected example
    question = st.text_area("Type your question here:", value=example)

    # Filters (if applicable)
    st.write("### Filters")
    filter_author = st.checkbox("Filter by Year", False)
    if filter_author:
        # Add logic to filter by author
        st.selectbox("Select an Author", ["2020", "2021", "2022"])  # Update with actual author names

    if st.button("Ask Bot"):
        if not question:
            st.error("Please enter a question.")
        else:
            # Run the chain and capture the output
            with st.spinner("Running the chain..."):
                result = chain.run(question)
            
            # Display the generated answer with a larger text area
            st.text_area("Answer:", result, height=300)