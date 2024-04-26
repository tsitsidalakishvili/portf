import os
import streamlit as st
import openai
from pathlib import Path
from moviepy.editor import VideoFileClip

# Try to import APIKEY from constants.py
try:
    from constants import APIKEY
    openai.api_key = APIKEY  # Use the imported APIKEY
except ImportError:
    APIKEY = None  # Set APIKEY to None if import fails

def ensure_directory_exists(directory):
    """Ensure that the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

@st.cache_data
def extract_audio(video_file_path, output_audio_path):
    """Extract audio from a video file and save it as an MP3 file in the specified directory."""
    video = VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path, codec='mp3')
    audio.close()
    video.close()



def ensure_directory_exists(directory):
    """Ensure that the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

@st.cache_data
def extract_audio(video_file_path, output_audio_path):
    """Extract audio from a video file and save it as an MP3 file in the specified directory."""
    video = VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path, codec='mp3')
    audio.close()
    video.close()

@st.cache_data
def transcribe(audio_file_path):
    """Transcribe the specified audio file using OpenAI's Whisper model."""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
            return transcription['text'] if 'text' in transcription else "No transcript available."
    except Exception as e:
        st.error(f"Failed to transcribe audio: {str(e)}")
        return ""




@st.cache_data
def summarize_transcription(transcription, context):
    """Summarize the transcription using OpenAI's language model with additional context, focusing on extracting actionable tasks and decisions."""
    messages = [
        {"role": "system", "content": f"Please summarize the following software development meeting transcription into key decisions, and important points. Context: {context}"},
        {"role": "user", "content": transcription}
    ]
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        try:
            st.write("Sending the following prompt to the AI for summarization:")
            for message in messages:
                if message['role'] == "system":
                    st.write(f"{message['role'].title()}: {message['content']}")
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.5
            )
            return response['choices'][0]['message']['content']
        except openai.error.APIConnectionError as e:
            attempts += 1
            st.error("Network error, trying again...")
            time.sleep(1)  # Wait a bit before retrying
        except openai.error.APIError as e:
            st.error(f"API error: {e}")
            return "Summarization failed due to API error."
        except Exception as e:
            st.error(f"Failed to summarize transcription: {str(e)}")
            return "Summarization failed."
        if attempts == max_attempts:
            st.error("Failed after multiple attempts, please try again later.")
            return "Summarization failed after multiple retries."



@st.cache_data
def generate_jira_items(summary, context=""):
    """Generate structured Jira issue breakdowns from the provided summary, organizing them into epics and tasks suitable for Agile development."""
    messages = [
        {"role": "system", "content": "From the summary provided, please generate a structured breakdown into epics and associated tasks. Each epic should clearly outline the main objectives, and tasks should detail the steps required to achieve these objectives."},
        {"role": "system", "content": f"Additional context: {context}" if context else ""},
        {"role": "user", "content": summary}
    ]
    try:
        st.write("Sending the following structured breakdown request to the AI:")
        for message in messages:
            if message['role'] == "system" and message['content']:
                st.write(f"{message['role'].title()}: {message['content']}")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5
        )
        # Assuming the AI responds with newline-separated tasks and epics
        items = response['choices'][0]['message']['content'].strip().split('\n')
        return items  # This will now return a list of epics and tasks formatted as requested
    except Exception as e:
        st.error(f"Failed to generate Jira items: {str(e)}")
        return ["Item generation failed."]






def main():
    st.title("From Audio to JIRA")
    st.subheader("Audio and Video Upload and Transcription App")
    temp_dir = r'C:\Users\dalak\OneDrive\Desktop\App\WIP\tempDir'
    ensure_directory_exists(temp_dir)  # Make sure the tempDir exists
    
    # Check if APIKEY was not imported
    if APIKEY is None:
        st.subheader("Enter your OpenAI API Key:")
        input_api_key = st.text_input("API Key")
        if input_api_key:
            openai.api_key = input_api_key
        else:
            st.stop()
    
    with st.expander("Upload Audio/Video"):
        uploaded_file = st.file_uploader("Choose a file", type=["mp3", "mp4", "m4a"])
        if uploaded_file is not None:
            file_name = os.path.join(temp_dir, uploaded_file.name)
            file_type = uploaded_file.type.split('/')[1]  # Extracts 'mp3', 'mp4', or 'm4a' from the MIME type
            
            with open(file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if file_type == "mp4":
                audio_file_path = file_name.split('.')[0] + '.mp3'
                extract_audio(file_name, audio_file_path)
                st.video(file_name)
            else:
                audio_file_path = file_name
                st.audio(file_name, format=f'audio/{file_type}')
    
    if uploaded_file is not None:
        with st.expander("Transcribe Audio"):
            if st.button("Start Transcription"):
                transcription = transcribe(audio_file_path)
                st.text_area("Transcription:", value=transcription, height=200)
                st.session_state.transcription = transcription  # Store transcription in session state

        if 'transcription' in st.session_state:
            with st.expander("Summarize Transcript"):
                summarization_context = st.text_input("Enter context for better summarization:", value="")
                summarize_button = st.button("Summarize")
                if summarize_button:
                    summary = summarize_transcription(st.session_state.transcription, summarization_context)
                    st.session_state.summary = summary  # Store summary in session state
                    st.text_area("Summary:", value=summary, height=200)

            # Ensure Jira Issues expander appears only after summary is available
            if 'summary' in st.session_state:
                with st.expander("Create Jira Issues"):
                    jira_context = st.text_input("Enter context for Jira issue breakdown:", value="")
                    if st.button("Generate Jira Issue Breakdown"):
                        jira_items = generate_jira_items(st.session_state.summary, jira_context)
                        st.write("Epics and Tasks for Jira:")
                        for item in jira_items:
                            if item:  # Check if the item is not an empty string
                                st.write(item)  # Displays each item as a separate bullet point or line


        os.remove(file_name)
        if file_type == "mp4":
            os.remove(audio_file_path)

if __name__ == "__main__":
    main()
