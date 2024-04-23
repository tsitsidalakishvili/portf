import streamlit as st
import pandas as pd
import requests
import openai
import json

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

# Function to initialize default prompt templates
# Function to initialize default prompt templates
def initialize_default_prompt_templates():
    templates = [
        {
            "name": "Issue Tracker",
            "instructions": "As a Project Coordinator, provide an integrated report on the status, recent developments, and any blockers or dependencies related to a specific issue.",
            "example_input": "Provide a detailed report on issue KEY-123.",
            "example_output": "Issue KEY-123, titled [Summary], is currently [Status]. The latest update was on [Updated], with [Comment count] comments, and the last comment: [Latest Comment]. There are [Blocker count] blockers impeding progress, with the main dependency on [Dependency Key].",
            "query_template": "Provide a detailed report on issue {issue_key}.",
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
    st.session_state.data['prompt_templates'] = templates if not st.session_state.data['prompt_templates'] else st.session_state.data['prompt_templates']

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
    




# Sidebar for User Input and Mode Selection
with st.sidebar:
    st.title("Settings")
    app_mode = st.selectbox('Choose the Mode', ['Home', 'Manage Data Sources', 'Manage Prompts'])

# Home Page
if app_mode == 'Home':
    st.markdown("""
    Streamlit Application for Data Analysis with LLM Integration

    This Streamlit application is designed to seamlessly integrate data from Jira, user-uploaded CSV files with the capabilities of Large Language Models (LLMs). 
    This facilitates data analysis and interaction through tailored prompts, making it easier to derive insights from various data sources.

    How It Works
    ------------

    - Data Source Integration:
        - Jira: Fetch projects and issues from your Jira account. Specify the project and customize the JQL query to retrieve relevant data.
        - Upload .csv: Easily upload your own datasets through CSV files for quick and direct analysis, enhancing flexibility in data integration.

    - Prompt Management:
        - Add/Edit Prompt Templates: Define templates with specific instructions, example inputs/outputs, query templates, and the number of few-shot examples. These guide the LLM in generating responses based on your data.
        - Execute Prompts: Utilize your templates to query the integrated data, receiving responses crafted by the LLM for dynamic interaction and insights.


    Tips for Effective Prompt Engineering
    -------------------------------------

    - Be Specific: Clearly define the role and knowledge scope you expect the LLM to assume in your prompts for more accurate and relevant responses.

    - Utilize Few-Shot Learning: Incorporate examples in your prompts to guide the LLM. Few-shot examples can significantly improve the model's understanding and output quality.

    - Dynamic Queries: Leverage the power of dynamic variables in your templates to make your prompts adaptable to different queries and data points.

    Explore the capabilities of connecting data to LLM for an enhanced data analysis experience.

    """)



# Data Source Management
if app_mode == 'Manage Data Sources':
    with st.sidebar:
        data_source = st.radio("Select Data Source", ['Upload CSV', 'Jira'])
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

    if not st.session_state.data['jira_df'].empty:
        st.title("Select Columns for Analysis")
        selected_columns = st.multiselect("Select columns to save for analysis:", st.session_state.data['jira_df'].columns)
        st.session_state.data['selected_columns'] = selected_columns



if app_mode == 'Manage Prompts':
    with st.sidebar:
        openai_api_key = st.text_input('OpenAI API key', type='password')
        if openai_api_key:
            st.session_state.data['openai_api_key'] = openai_api_key
            openai.api_key = openai_api_key

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

# Sidebar
with st.sidebar:    
    # Add a link to your LinkedIn profile
    st.markdown("""
        <hr style="border:1px solid #f0f2f6;"> <!-- Optional: This adds a horizontal line for visual separation -->
        Created by <a href="https://www.linkedin.com/in/tsitsi-dalakishvili" target="_blank">Tsitsi Dalakishvili</a>
        """, unsafe_allow_html=True)