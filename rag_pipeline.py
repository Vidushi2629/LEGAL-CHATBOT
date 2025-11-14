from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from gtts import gTTS
import os

# Load environment variables
load_dotenv()

# Setup Groq LLM
llm_model = ChatGroq(model="openai/gpt-oss-120b")

#  Generate context
def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# Template for answers
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know — don’t try to make up an answer.
Only use the information from the provided context.

Question: {question}
Context: {context}
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    result = chain.invoke({"question": query, "context": context})
    return result.content if hasattr(result, "content") else str(result)

#  Multi-perspective summaries
student_prompt = """
Summarize the following case for a *law student*.
Explain facts, evidence, reasoning, and judgment in simple educational language.
Context:
{context}
"""

lawyer_prompt = """
Summarize the following case for a *lawyer*.
Highlight key legal points, arguments, precedents, and strategy angles.
Context:
{context}
"""

judge_prompt = """
Summarize the following case for a *judge*.
Focus on admissible evidence, charges, and judicial reasoning objectively.
Context:
{context}
"""

def generate_summary(documents, model, user_type):
    context = get_context(documents)
    if user_type == "student":
        prompt = ChatPromptTemplate.from_template(student_prompt)
    elif user_type == "lawyer":
        prompt = ChatPromptTemplate.from_template(lawyer_prompt)
    else:
        prompt = ChatPromptTemplate.from_template(judge_prompt)
    chain = prompt | model
    result = chain.invoke({"context": context})
    return result.content if hasattr(result, "content") else str(result)

#  Generate TTS audio file (for Streamlit playback)
def generate_audio_file(text):
    try:
        tts = gTTS(text)
        audio_path = "response_audio.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print("Error generating audio:", e)
        return None
