import streamlit as st
import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources are downloaded
nltk.download('stopwords')

# Initialize session state for storing data and configurations
if 'data' not in st.session_state:
    st.session_state.data = {
        'uploaded_df': pd.DataFrame(),
        'jira_df': pd.DataFrame(),
        'jira_domain': '',
        'jira_username': '',
        'jira_api_token': '',
        'selected_columns': []
    }

# Define function to preprocess and calculate text similarity
def preprocess_data(dataframe, text_column):
    stop_words = set(stopwords.words('english'))
    dataframe['clean_text'] = dataframe[text_column].apply(lambda x: ' '.join([word for word in str(x).lower().split() if word not in stop_words]))
    dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', x))
    dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: re.sub(r'http\S+|www\S+', '', x))
    dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: re.sub(r'<.*?>', '', x))
    return dataframe

def calculate_similarity(df, threshold, identifier_column, text_column, additional_columns):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                pairs.append((df[identifier_column].iloc[i], df[identifier_column].iloc[j], similarity_matrix[i, j]))
    pairs_df = pd.DataFrame(pairs, columns=['Node 1', 'Node 2', 'Similarity'])
    for col in additional_columns:
        pairs_df = pairs_df.merge(df[[identifier_column, col]], left_on='Node 1', right_on=identifier_column).drop(columns=[identifier_column]).rename(columns={col: f'{col}_Node1'})
        pairs_df = pairs_df.merge(df[[identifier_column, col]], left_on='Node 2', right_on=identifier_column).drop(columns=[identifier_column]).rename(columns={col: f'{col}_Node2'})
    return pairs_df


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

def similarity_func(df):
    with st.expander("Similarity Functionality"):
        st.subheader("Similarity Results")

        # Use session state to retrieve columns, remove the 'Save Columns' button
        text_column = st.session_state.get('text_column', df.columns[0])
        identifier_column = st.session_state.get('identifier_column', df.columns[0])
        additional_columns = st.session_state.get('additional_columns', df.columns[0])

        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.05)

        if st.button('Start Similarity Analysis'):
            # Check if columns exist
            if set([st.session_state.text_column, st.session_state.identifier_column] + st.session_state.additional_columns).issubset(set(df.columns)):
                try:
                    # Ensure text_column is of string type
                    df[st.session_state.text_column] = df[st.session_state.text_column].astype(str)
                    
                    # Preprocess and calculate similarity
                    preprocessed_data = preprocess_data(df, st.session_state.text_column)
                    similar_pairs = calculate_similarity(df, threshold, st.session_state.identifier_column, st.session_state.text_column, st.session_state.additional_columns)

                    # Diagnostic outputs
                    st.write(f"Number of rows in the original data: {len(preprocessed_data)}")
                    st.write(f"Number of similar pairs found: {len(similar_pairs)}")

                    # Display similarity results
                    st.subheader(f"Similarity Threshold: {threshold}")
                    st.dataframe(similar_pairs)
                except Exception as e:
                    st.error(f"Error running similarity analysis. Error: {str(e)}")
            else:
                st.error("Selected columns are not present in the data. Please check the column names and try again.")


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

# Streamlit UI setup
st.title('Text Similarity Analysis')

# Sidebar for mode selection
with st.sidebar:
    app_mode = st.selectbox('Choose the Mode', ['How it works', 'Manage Data Sources', 'Calculate Similarity'])

# Home Page with Description
if app_mode == 'How it works':
    st.markdown("""
    ## Text Similarity Analysis with Streamlit
    
    This application allows you to analyze text data for similarity. You can upload your data, connect to external sources like Jira, and calculate similarity scores based on text content. The application uses TF-IDF for vectorization and cosine similarity to determine how similar texts are to each other.
    
    ### Features:
    - **Manage Data Sources**: Upload your CSV files or connect to Jira to fetch issues.
    - **Calculate Similarity**: Select text columns and calculate similarity scores between texts.
    """)

# Data Source Management
if app_mode == 'Manage Data Sources':
    with st.sidebar:
        data_source = st.radio("Select Data Source", ['Upload CSV', 'Jira', 'Neo4j'])



    # Correct the handling of uploaded CSV data:
    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
        if uploaded_file is not None:
            try:
                # Update this line to correct the key
                st.session_state.data['uploaded_df'] = pd.read_csv(uploaded_file)
                st.success('CSV file uploaded successfully!')
                st.write(st.session_state.data['uploaded_df'])  # Ensure this references 'uploaded_df'
            except Exception as e:
                st.error(f'Failed to read CSV file: {e}')
        else:
            st.info("Upload a CSV file to proceed.")



    # Correct the handling of Jira fetched data:
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
        uri = st.sidebar.text_input("Neo4j URI", "bolt://localhost:7687")
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





    if not st.session_state.data['jira_df'].empty:
        st.title("Select Columns for Analysis")
        selected_columns = st.multiselect("Select columns to save for analysis:", st.session_state.data['jira_df'].columns)
        st.session_state.data['selected_columns'] = selected_columns




# In the "Calculate Similarity" mode:
elif app_mode == 'Calculate Similarity':
    # Ensure this references the unified DataFrame key
    df = st.session_state.data.get('uploaded_df', pd.DataFrame())
    if not df.empty:
        st.write(df)
        text_column = st.selectbox("Choose Text Column for Analysis", df.columns)
        identifier_column = st.selectbox("Choose Identifier Column", df.columns)
        additional_columns = st.multiselect("Choose Additional Columns (for context)", df.columns)
        threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        if st.button("Calculate Similarity"):
            processed_df = preprocess_data(df, text_column)
            results_df = calculate_similarity(processed_df, threshold, identifier_column, 'clean_text', additional_columns)
            if not results_df.empty:
                st.dataframe(results_df)
            else:
                st.write("No similar items found with the current threshold.")

    
    if all(key in st.session_state for key in ['text_column', 'identifier_column', 'additional_columns']):
        similarity_func(df)