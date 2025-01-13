"""Utilities for processing and analyzing text from legal documents."""

from typing import List, Optional
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from ..config.analysis_config import AnalysisMode, AnalysisConfig, ANALYSIS_CONFIGS

def create_text_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Split text into manageable chunks for processing.
    
    Args:
        text: Raw text to be split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_conversation_chain(
    text_chunks: List[str],
    config: AnalysisConfig
) -> ConversationalRetrievalChain:
    """
    Create a conversation chain for document analysis.
    
    Args:
        text_chunks: Processed text chunks
        config: Analysis configuration
    
    Returns:
        Configured conversation chain
    """
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    # Initialize language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=config.temperature,
        convert_system_message_to_human=True
    )
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=config.prompt_template
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    # Create and return the conversation chain
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

def process_document(
    text: str,
    mode: AnalysisMode
) -> Optional[ConversationalRetrievalChain]:
    """
    Process a document using specified analysis mode.
    
    Args:
        text: Document text
        mode: Analysis mode to use
    
    Returns:
        Configured conversation chain or None if processing fails
    """
    try:
        config = ANALYSIS_CONFIGS[mode]
        chunks = create_text_chunks(
            text,
            config.chunk_size,
            config.chunk_overlap
        )
        return create_conversation_chain(chunks, config)
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return None
