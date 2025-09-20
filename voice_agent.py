import os
import time
import sounddevice as sd
import numpy as np
import pyttsx3
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# --- 1. Configuration and Setup ---

# Load environment variables from .env file
load_dotenv()

# Whisper STT Configuration
MODEL_SIZE = "base.en"      # "tiny.en", "base.en", "small.en", "medium.en"
WHISPER_DEVICE = "cpu"      # "cpu" or "cuda" if you have an NVIDIA GPU
WHISPER_COMPUTE = "int8"    # "int8" for CPU, "float16" for GPU

# Audio Recording Configuration
# MODIFICATION 1: Increased listening window to 10 seconds
INTERVAL = 10               # Record in 10-second chunks
SAMPLE_RATE = 16000

# --- 2. Initialize Core Components ---

print("Initializing components...")

# Initialize Text-to-Speech Engine with better error handling
try:
    tts_engine = pyttsx3.init()
    print("✅ TTS engine initialized successfully.")
except Exception as e:
    print(f"❌ CRITICAL: Error initializing TTS engine: {e}")
    print("   The agent will not be able to speak. Please check your system's TTS drivers.")
    tts_engine = None

# Initialize Whisper Model
print(f"Loading Whisper model '{MODEL_SIZE}'...")
whisper_model = WhisperModel(MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print("✅ Whisper model loaded.")

# Initialize LangChain Agent
llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def get_current_time_and_location():
    """Returns the current time and location."""
    current_time = time.strftime("%A, %B %d, %Y %I:%M %p")
    return f"The current time in Varanasi, India is {current_time}."

tools = [get_current_time_and_location]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful voice assistant. Keep your responses concise and conversational, suitable for a voice bot. The current location is Varanasi, India."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

print("✅ All components initialized. The agent is ready.")

# --- 3. Helper Functions ---

def speak(text):
    """Converts text to speech and gives terminal feedback."""
    if tts_engine and text:
        print(f"🤖 AGENT: {text}")
        print("Speaking...")
        # This function blocks until the speech is finished
        tts_engine.say(text)
        tts_engine.runAndWait()
        print("Finished speaking.")
    elif not tts_engine:
        print("묵 TTS engine not available. Cannot speak.")
    else:
        # This case is for when text is empty
        pass


# --- 4. Main Application Loop ---

def main():
    """The main loop to listen, transcribe, think, and speak."""
    print("\n--- Voice Agent Activated ---")
    speak("Hello! I'm now online. How can I help you?")
    
    # MODIFICATION 2: Add a 4-second pause after the greeting
    print("(Pausing for 4 seconds before listening...)")
    time.sleep(4)

    while True:
        try:
            # Listen to the microphone for the specified interval
            print(f"\n🎧 Listening for {INTERVAL} seconds...")
            audio_data = sd.rec(int(INTERVAL * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()

            print("📝 Transcribing...")
            segments, _ = whisper_model.transcribe(audio_data.flatten(), beam_size=5, vad_filter=True)
            
            user_text = "".join(segment.text for segment in segments).strip()

            if not user_text or len(user_text) < 2:
                print("(No clear speech detected)")
                continue

            print(f"👤 YOU: {user_text}")

            print("🤔 Thinking...")
            response = agent_executor.invoke({"input": user_text})
            agent_response = response.get("output", "I'm not sure how to respond to that.")
            
            # Speak the response
            speak(agent_response)
            
            # MODIFICATION 2: Add a 4-second pause after the AI speaks
            print("(Pausing for 4 seconds before listening again...)")
            time.sleep(4)

        except KeyboardInterrupt:
            print("\nShutting down agent.")
            speak("Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()