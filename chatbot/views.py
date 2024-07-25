import os
import asyncio
import base64
import io
import time
import re
from PIL import Image
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from django.shortcuts import render
from django.http import JsonResponse
from asgiref.sync import sync_to_async
from datasets import load_dataset
from gtts import gTTS
from django.conf import settings

# Set the API keys directly
groq_api_key = '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'
google_api_key = 'AIzaSyAjZY4oublS-3OjdZ7cGm5L5xgUVFBf2M4'
huggingface_api_key = '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'

# Initialize Groq Langchain chat object
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')

# Configure Google Generative AI
genai.configure(api_key=google_api_key)
model_name = next((m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods), None)
if not model_name:
    raise ValueError("No suitable model found.")

llm_google = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key)
llm_google_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

# Load the Indian legal corpus dataset
ds = load_dataset("sujantkumarkv/indian_legal_corpus", use_auth_token=huggingface_api_key)

# Define the prompts
groq_system_prompt = """
You are an expert in Indian law. Provide brief and concise legal advice, mentioning relevant acts and examples. Keep your responses short and to the point.
"""

google_system_prompt = """
You are an Indian law expert named Nivan. Provide brief and concise legal advice, mentioning relevant acts and examples. Keep your responses short and to the point.
"""

# Define the prompt template for Groq
groq_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=groq_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

# Initialize conversational memory
memory = ConversationBufferWindowMemory(k=8, memory_key="chat_history", return_messages=True)

# Create a conversation chain for Groq
groq_conversation = LLMChain(
    llm=groq_chat,
    prompt=groq_prompt_template,
    verbose=False,
    memory=memory,
)

# Initialize conversation history
conversation_history = []

# Function to clean response text
def clean_response(text):
    cleaned_text = text.replace('*', '')
    cleaned_text = '\n'.join(line.strip() for line in cleaned_text.split('\n'))
    return cleaned_text

# Updated ask_groq function
async def ask_groq(question, chat_history):
    memory.clear()
    for message in chat_history[-5:]:  # Consider only the last 5 messages
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )
    try:
        response = await sync_to_async(groq_conversation.predict)(human_input=question)
        return clean_response(response)
    except Exception as e:
        return clean_response(f"An error occurred: {str(e)}")

# Updated ask_google function
async def ask_google(question, conversation_history):
    history_text = "\n".join([f"Human: {msg['human']}\nAI: {msg['AI']}" for msg in conversation_history[-5:]])
    prompt = f"""
    {google_system_prompt}
    Conversation History:
    {history_text}
    
    Human: {question}
    AI:
    """
    result = await sync_to_async(llm_google.invoke)(prompt)
    return clean_response(result.content)

# Function to process image and ask question to Google Generative AI Vision
async def process_image_question(image, question):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_data_uri = f"data:image/jpeg;base64,{img_str}"
    
    prompt = f"""You are Nivan, an AI bot with extensive knowledge of Indian law. 
    Analyze the image and answer the following question: "{question}"
    Provide a detailed explanation, drawing on your expertise in Indian law where relevant. 
    Be thorough in your analysis and explain any legal concepts or implications you observe."""
    
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },
            {"type": "image_url", "image_url": img_data_uri},
        ]
    )
    result = await sync_to_async(llm_google_vision.invoke)([message])
    return clean_response(result.content)

# Function to get additional information from the dataset
def get_dataset_info(question):
    relevant_entries = [entry for entry in ds['train'] if question.lower() in entry['text'].lower()]
    if relevant_entries:
        return clean_response(relevant_entries[0]['text'])
    return clean_response("Any further Questions ? I am happy to help you !")

# Function to combine responses with dataset information
def combine_responses(gemini_response, groq_response, dataset_info):
    combined_response = f"{gemini_response}\n\n{groq_response}\n\n{dataset_info}"
    return clean_response(combined_response)

# Text-to-speech function
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    filename = f"response_{int(time.time())}.mp3"
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    tts.save(file_path)
    return os.path.join(settings.MEDIA_URL, filename)

# Function to get predefined responses
def get_predefined_response(message):
    message = message.lower()
    
    identity_questions = ["who are you", "what are you", "who made you", "who created you"]
    if any(question in message for question in identity_questions):
        return ("I'm Nivan, the AI helper created by six undergrads. I'm intended to assist with a wide range of task "
                "discussions concerning Indian legal concerns. I do not have a physical form or avatar; instead, I am "
                "a model of language trained to engage in conversation and aid with chores. How can I help you today?")
    
    if "how are you" in message:
        return "I am fine! What about you? How can I help you regarding any Legal issues?"
    
    if "what are you doing" in message:
        return "Ready to solve your Legal issues! Any questions?"
    
    goodbyes = ["bye", "goodbye", "exit", "see you", "farewell"]
    if any(goodbye in message for goodbye in goodbyes):
        return "Goodbye! Feel free to return if you have any legal issues, questions, or need assistance in the future."
    
    return None

# Updated chatbot view
async def chatbot(request):
    global conversation_history
    
    if request.method == 'POST':
        message = request.POST.get('message')
        
        # Check for predefined responses
        predefined_response = get_predefined_response(message)
        if predefined_response:
            conversation_history.append({'human': message, 'AI': predefined_response})
            audio_file = text_to_speech(predefined_response)
            return JsonResponse({
                'message': message, 
                'response': predefined_response,
                'audio_url': request.build_absolute_uri(audio_file)
            })
        
        # Check if an image was uploaded
        uploaded_image = request.FILES.get('image')
        
        if uploaded_image:
            # Process the image and question
            image = Image.open(uploaded_image)
            response = await process_image_question(image, message)
        else:
            # Concurrently get responses from both APIs
            gemini_response, groq_response = await asyncio.gather(
                ask_google(message, conversation_history),
                ask_groq(message, conversation_history)
            )
            
            # Get additional information from the dataset
            dataset_info = get_dataset_info(message)
            
            # Combine responses
            response = combine_responses(gemini_response, groq_response, dataset_info)
        
        # Update conversation history
        conversation_history.append({'human': message, 'AI': response})
        
        # Limit conversation history to last 10 messages
        conversation_history = conversation_history[-10:]
        
        # Generate audio file
        audio_file = text_to_speech(response)
        
        return JsonResponse({
            'message': message, 
            'response': response,
            'audio_url': request.build_absolute_uri(audio_file)
        })
    
    return render(request, 'chatbot.html') 
