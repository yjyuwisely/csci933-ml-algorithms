import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads



os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ.pop('LANGCHAIN_ENDPOINT', None)
os.environ.pop('LANGCHAIN_API_KEY', None)
os.environ['USER_AGENT'] = 'DEMO'

# Load local Chroma vectorstore with HuggingFace embeddings
vectorstore = Chroma(persist_directory="../chroma_db", embedding_function=HuggingFaceEmbeddings()) 
retriever = vectorstore.as_retriever()
# Load the Mistral model from Ollama
llm = OllamaLLM(model="mistral", base_url="http://localhost:11500")

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def rag_fusion_retriever(query: str, n: int = 0) -> list:
    """
    Generates n questions to the given query using RAG fusion with vectorstore and LLM.
    """
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
    )
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    docs = retrieval_chain_rag_fusion.invoke({"question": query})
    docs = [doc for doc, _ in docs]  # Extract only the documents from the tuples
    if n > 0:
        docs = docs[:n]  # Limit to n documents if specified
    return docs

def rag_fusion_generate_answer(query: str) -> str:
    """
    Generates an answer to the given query using the vectorstore and LLM.
    """
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Retrieve documents and combine their content into a single string
    docs = rag_fusion_retriever(query)
    context = "\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke({"question": query, "context": context})

# def rag_fusion_generate_answer(query: str) -> str:
#     """
#     Generates an answer to the given query using the vectorstore and LLM.
#     """
#     template = """Answer the question based only on the following context:
#     {context}

#     Question: {question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)
#     rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
#     )
#     return rag_chain.invoke({"question": query, "context": rag_fusion_retriever(query)})
