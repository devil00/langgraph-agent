"""LangGraph Agent"""
import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState

from langgraph.prebuilt import tools_condition  # type: ignore[import-not-found]
from langgraph.prebuilt import ToolNode  # type: ignore[import-not-found]
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client

import tempfile
import requests
import whisper
import imageio
import yt_dlp

from PIL import Image
from typing import List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from smolagents import tool, LiteLLMModel
import google.generativeai as genai
from pytesseract import image_to_string

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}



#  YouTube Frame Sampler 
@tool
def youtube_frames_to_images(url: str, every_n_seconds: int = 5) -> List[Image.Image]:
    """
    Downloads a YouTube video and extracts frames at regular intervals.

    Args:
        url (str): The URL of the YouTube video to process.
        every_n_seconds (int): The time interval in seconds between extracted frames.

    Returns:
        List[Image.Image]: A list of sampled frames as PIL images.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_cfg = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": os.path.join(temp_dir, "yt_video.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "force_ipv4": True
        }
        with yt_dlp.YoutubeDL(ydl_cfg) as ydl:
            ydl.extract_info(url, download=True)

        video_file = next((os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.mp4')), None)
        reader = imageio.get_reader(video_file)
        fps = reader.get_meta_data().get("fps", 30)
        interval = int(fps * every_n_seconds)

        return [Image.fromarray(frame) for i, frame in enumerate(reader) if i % interval == 0]


#  File Reading Tool 
@tool
def read_text_file(file_path: str) -> str:
    """
    Reads plain text content from a file.

    Args:
        file_path (str): The full path to the text file.

    Returns:
        str: The contents of the file, or an error message.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


#  File Downloader 
@tool
def file_from_url(url: str, save_as: Optional[str] = None) -> str:
    """
    Downloads a file from a URL and saves it locally.

    Args:
        url (str): The URL of the file to download.
        save_as (Optional[str]): Optional filename to save the file as.

    Returns:
        str: The local file path or an error message.
    """
    try:
        if not save_as:
            parsed = urlparse(url)
            save_as = os.path.basename(parsed.path) or f"file_{os.urandom(4).hex()}"

        file_path = os.path.join(tempfile.gettempdir(), save_as)
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        return f"File saved to {file_path}"
    except Exception as e:
        return f"Download failed: {e}"


#  Audio Transcription (YouTube) 
@tool
def transcribe_youtube(yt_url: str) -> str:
    """
    Transcribes the audio from a YouTube video using Whisper.

    Args:
        yt_url (str): The URL of the YouTube video.

    Returns:
        str: The transcribed text of the video.
    """
    model = whisper.load_model("small")

    with tempfile.TemporaryDirectory() as tempdir:
        ydl_opts = {
            "format": "bestaudio",
            "outtmpl": os.path.join(tempdir, "audio.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav"
            }],
            "quiet": True,
            "force_ipv4": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(yt_url, download=True)

        wav_file = next((os.path.join(tempdir, f) for f in os.listdir(tempdir) if f.endswith(".wav")), None)
        return model.transcribe(wav_file)['text']


#  Audio File Transcriber 
@tool
def audio_to_text(audio_path: str) -> str:
    """
    Transcribes an uploaded audio file into text using Whisper.

    Args:
        audio_path (str): The local file path to the audio file.

    Returns:
        str: The transcribed text or an error message.
    """
    try:
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        return f"Failed to transcribe: {e}"


#  OCR 
@tool
def extract_text_via_ocr(image_path: str) -> str:
    """
    Extracts text from an image using Optical Character Recognition (OCR).

    Args:
        image_path (str): The local path to the image file.

    Returns:
        str: The extracted text or an error message.
    """
    try:
        img = Image.open(image_path)
        return image_to_string(img)
    except Exception as e:
        return f"OCR failed: {e}"


#  CSV Analyzer 
@tool
def summarize_csv_data(path: str, query: str = "") -> str:
    """
    Provides a summary of the contents of a CSV file.

    Args:
        path (str): The file path to the CSV file.
        query (str): Optional query to run on the data.

    Returns:
        str: Summary statistics and column details or an error message.
    """
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return f"Loaded CSV with {len(df)} rows. Columns: {list(df.columns)}\n\n{df.describe()}"
    except Exception as e:
        return f"CSV error: {e}"


#  Excel Analyzer 
@tool
def summarize_excel_data(path: str, query: str = "") -> str:
    """
    Provides a summary of the contents of an Excel file.

    Args:
        path (str): The file path to the Excel file (.xls or .xlsx).
        query (str): Optional query to run on the data.

    Returns:
        str: Summary statistics and column details or an error message.
    """
    try:
        import pandas as pd
        df = pd.read_excel(path)
        return f"Excel file with {len(df)} rows. Columns: {list(df.columns)}\n\n{df.describe()}"
    except Exception as e:
        return f"Excel error: {e}"



# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)

# build a retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #  dim=768
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"), 
    os.environ.get("SUPABASE_SERVICE_KEY"))
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding= embeddings,
    table_name="documents",
    query_name="match_documents_langchain_2",
)
create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)



tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    web_search,
    arvix_search,
    youtube_frames_to_images,
    read_text_file,
    file_from_url,
    transcribe_youtube,
    audio_to_text,
    extract_text_via_ocr,
    summarize_csv_data,
    summarize_excel_data,
]

# Build graph function
def build_graph(provider: str = "huggingface"):
    """Build the graph"""
    # Load environment variables from .env file
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        # Groq https://console.groq.com/docs/models
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0) # optional : qwen-qwq-32b gemma2-9b-it
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()

# test
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # Build the graph
    graph = build_graph(provider="huggingface")
    # Run the graph
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
