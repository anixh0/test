import os
import asyncio
import base64
import io
import time
from PIL import Image
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from django.shortcuts import render
from django.http import JsonResponse
from asgiref.sync import sync_to_async
from datasets import load_dataset
from gtts import gTTS
from django.conf import settings
import aiohttp
import aiofiles
from functools import lru_cache

# Set the API keys directly
groq_api_key = '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'
google_api_key = 'AIzaSyBsNsY1-gm3D2INK1TJKpgbm-YPc6SxpWg'
huggingface_api_key = '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'

# Initialize Groq Langchain chat object with Llama-3.1-70b-Versatile
groq_chat = ChatGroq(api_key=groq_api_key, model_name='llama-3.1-70b-versatile')

# Configure Google Generative AI for image processing only
genai.configure(api_key=google_api_key)
llm_google_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

# Load the Indian legal corpus dataset
ds = load_dataset("sujantkumarkv/indian_legal_corpus", use_auth_token=huggingface_api_key)

# Define the prompts
groq_system_prompt = """
You are Nivan, an expert in Indian law, with knowledge up to 2024. Provide legal advice in the following structure:
1. Explanation: Briefly explain the legal concept or issue.
2. Relevant Law: Cite the specific Indian law act or section that applies.
3. Example: Provide a practical example to illustrate the application of the law.

Keep your responses concise and to the point.
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

# Updated ask_groq function to use aiohttp for async API calls
async def ask_groq(question, chat_history):
    memory.clear()
    for message in chat_history[-5:]:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-70b-versatile",
                    "messages": [
                        {"role": "system", "content": groq_system_prompt},
                        {"role": "user", "content": question}
                    ]
                }
            ) as resp:
                result = await resp.json()
                response = result['choices'][0]['message']['content']
        return clean_response(response)
    except Exception as e:
        return clean_response(f"An error occurred: {str(e)}")

# Function to process image and ask question to Google Generative AI Vision
async def process_image_question(image, question):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_data_uri = f"data:image/jpeg;base64,{img_str}"
    
    prompt = f"""You are Nivan, an AI bot with extensive knowledge of Indian law up to 2024. 
    Analyze the image and answer the following question: "{question}"
    Provide a response in the following structure:
    1. Explanation: Briefly explain what you see in the image related to the question.
    2. Relevant Law: Cite any specific Indian law act or section that might apply to the situation in the image.
    3. Example: Provide a practical example of how the law might apply in a similar real-world scenario."""
    
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

# Function to get additional information from the dataset with caching
@lru_cache(maxsize=100)
def get_dataset_info(question):
    relevant_entries = [entry for entry in ds['train'] if question.lower() in entry['text'].lower()]
    if relevant_entries:
        return clean_response(relevant_entries[0]['text'])
    return ""

# Function to combine responses with dataset information
def combine_responses(groq_response, dataset_info):
    if dataset_info:
        combined_response = f"{groq_response}\n\nAdditional Information:\n{dataset_info}"
    else:
        combined_response = groq_response
    return clean_response(combined_response)

# Asynchronous text-to-speech function
async def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    filename = f"response_{int(time.time())}.mp3"
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    await sync_to_async(tts.save)(file_path)
    return os.path.join(settings.MEDIA_URL, filename)

# Function to get predefined responses
def get_predefined_response(message):
    message = message.lower()
    
    identity_questions = ["who are you", "what are you", "who made you", "who created you"]
    if any(question in message for question in identity_questions):
        return ("I'm Nivan, an AI assistant specializing in Indian law, with knowledge up to 2024. "
                "I was created to help with discussions and queries related to Indian legal matters. "
                "How can I assist you today?")
    
    if "how are you" in message:
        return "I'm functioning well, thank you! How can I assist you with any legal issues today?"
    
    if "what are you doing" in message:
        return "I'm here and ready to help with any legal questions or issues you might have. What can I do for you?"
    
    goodbyes = ["bye", "goodbye", "exit", "see you", "farewell"]
    if any(goodbye in message for goodbye in goodbyes):
        return "Thank you for using my services. If you have any more legal questions in the future, don't hesitate to ask. Goodbye!"
    
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
            audio_file = await text_to_speech(predefined_response)
            return JsonResponse({
                'message': message, 
                'response': predefined_response,
                'audio_url': request.build_absolute_uri(audio_file)
            })
        
        # Check if an image was uploaded
        uploaded_image = request.FILES.get('image')
        
        if uploaded_image:
            # Process the image and question using Gemini
            image = Image.open(uploaded_image)
            response = await process_image_question(image, message)
        else:
            # Get response from Groq
            groq_response = await ask_groq(message, conversation_history)
            
            # Get additional information from the dataset
            dataset_info = await sync_to_async(get_dataset_info)(message)
            
            # Combine responses
            response = combine_responses(groq_response, dataset_info)
        
        # Update conversation history
        conversation_history.append({'human': message, 'AI': response})
        
        # Limit conversation history to last 10 messages
        conversation_history = conversation_history[-10:]
        
        # Generate audio file asynchronously
        audio_file = await text_to_speech(response)
        
        return JsonResponse({
            'message': message, 
            'response': response,
            'audio_url': request.build_absolute_uri(audio_file)
        })
    
    return render(request, 'chatbot.html')


