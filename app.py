import os
import streamlit as st
from google import genai
from google.genai import types
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import io

# 1. Page Configuration & Title
st.set_page_config(page_title="Voice Bot", page_icon="🎙️")
st.title("🎙️ My Voice Bot")
st.write("Talk to an AI version of me! Record your question below.")

# 2. API Key Setup
# Securely fetch API key from Streamlit secrets or an environment variable
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.warning("Please configure your GEMINI_API_KEY in your environment variables or Streamlit secrets.")
    api_key = st.text_input("Or enter your Gemini API Key here:", type="password")

# 3. Define Your Personal Profile (Customize This!)
# The AI uses this data to answer questions exactly as you would.
MY_PROFILE = """
You are an AI clone of Anjana Paluru. You must respond to all questions in the first person ("I", "me", "my") as if you are the user themselves.

Here is the truth about you to answer the testing questions:
1. Life Story:  I am an AI Architect with 7 years of background in HR Tech. I hold an IT B.Tech from JNTU University. I build autonomous multi-agent systems and deep learning pipelines.
2. #1 Superpower: Bridging the gap between deep AI engineering and HR business strategy.
3. Top 3 Growth Areas: Orchestrating multi-agent networks, securing enterprise MCP environments, and scaling self-healing infrastructure.
4. Coworker Misconception:  People assume I only handle administrative HR tasks. In reality, I write neural networks and manage LLMOps pipelines.
5. Pushing Boundaries: I build zero-intervention autonomous agent swarms. I also train high-precision predictive compensation deep learning models.

Keep your answers conversational, concise, and professional.
"""
# Initialize Gemini Client if key is available
if api_key:
    client = genai.Client(api_key=api_key)

    # 4. Audio Input Component
    st.write("### 1. Speak to the Bot")
    audio_record = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key='recorder'
    )

    if audio_record:
        # Get audio bytes from the recorder
        audio_bytes = audio_record['bytes']
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("Processing your voice and thinking..."):
            try:
                # 5. Send Audio Directly to Gemini for Transcription & Processing
                # Gemini can natively understand audio files without a separate speech-to-text tool
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type="audio/wav",
                        ),
                        f"{MY_PROFILE}\n\nListen to the user's voice input above. Transcribe their question, and provide your spoken response based on my profile."
                    ]
                )
                
                output_text = response.text
                st.write(f"**AI Response:** {output_text}")

                # 6. Text-to-Speech Output
                # Convert Gemini's text response back into speech
                tts = gTTS(text=output_text, lang='en', slow=False)
                
                # Save speech to an in-memory buffer
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                st.write("### 2. Listen to Response")
                st.audio(audio_buffer, format="audio/mp3", autoplay=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
