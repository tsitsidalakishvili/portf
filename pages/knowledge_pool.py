import streamlit as st
import pandas as pd
import numpy as np

# Streamlit UI setup for Knowledge Pool Analysis
st.title('Knowledge Pool')

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload your JIRA CSV file here", type='csv')

if uploaded_file is not None:
    try:
        # Attempt to read the uploaded file with UTF-8 encoding
        data = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # Attempt with ISO-8859-1 encoding if UTF-8 fails
            data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            # Attempt with Windows-1252 encoding if ISO-8859-1 also fails
            data = pd.read_csv(uploaded_file, encoding='cp1252')
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        # Stop execution if unable to read the file
        st.stop()

    # Display the uploaded DataFrame
    st.write("Uploaded JIRA Data:")
    st.dataframe(data)

    # User input for analysis parameters
    task_column = st.selectbox("Select the column containing task descriptions:", data.columns)
    assignee_column = st.selectbox("Select the column containing assignee names:", data.columns)
    story_points_column = st.selectbox("Select the column containing story points for tasks:", data.columns)

    # Button to trigger analysis
    if st.button("Analyze Knowledge Pool"):
        # Processing data for knowledge pool
        knowledge_data = data.groupby(assignee_column).agg(
            Tasks_Completed=pd.NamedAgg(column=task_column, aggfunc='count'),
            Total_Story_Points_Delivered=pd.NamedAgg(column=story_points_column, aggfunc='sum'),
            Average_Story_Points_per_Task=pd.NamedAgg(column=story_points_column, aggfunc=lambda x: np.round(np.mean(x), 2))
        ).reset_index()

        # Displaying results
        st.subheader("Knowledge Pool Analysis Results")
        st.write("Summary of assignees' knowledge and skills based on task completion and story points:")
        st.dataframe(knowledge_data)

        # Optional: Detailed view per assignee
        detailed_assignee = st.selectbox("Select an assignee to view detailed task list:", options=knowledge_data[assignee_column])
        if detailed_assignee:
            detailed_tasks = data[data[assignee_column] == detailed_assignee]
            st.write(f"Tasks completed by {detailed_assignee}:")
            st.dataframe(detailed_tasks[[task_column, story_points_column]])
else:
    st.info("Please upload a CSV file to begin analysis.")
