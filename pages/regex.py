from langchain.llms import OpenAI
import streamlit as st
import constants

# Initialize the OpenAI model with Langchain
openai_model = OpenAI(api_key=constants.APIKEY)

def generate_regex_from_description_direct(description):
    # Format the prompt as a string
    prompt = f"Generate a regex pattern based on the following description: {description}"
    
    # Use the invoke method correctly with the string prompt
    response = openai_model.invoke(input=prompt)
    
    # Assuming Langchain's response is directly usable, adjust based on actual response format
    # This step needs adjustment based on how Langchain structures its response
    # The following line is speculative and should be adjusted to match actual response handling
    regex_pattern = response  # Adjust this line based on the actual structure of Langchain's response

    return regex_pattern

# Streamlit UI code for input and button
selection = st.radio("Select Option", ["Regex Generator", "Extract Text"], horizontal=True)

if selection == "Regex Generator":
    st.subheader("Generate your Regex Pattern")
    description = st.text_area("Enter your description in natural language:", "Example: A pattern that matches an email address")
    
    if st.button('Generate Regex Pattern'):
        regex_pattern = generate_regex_from_description_direct(description)
        if regex_pattern:
            st.success("Generated Regex Pattern:")
            st.code(regex_pattern)
        else:
            st.error("Failed to generate a regex pattern. Please try again.")
