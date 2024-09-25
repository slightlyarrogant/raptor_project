from langchain import hub
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.utils.config import VERBOSE
from src.inference.query import similarity_search

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vectorstore, embd, model):
    """
    Create a Retrieval-Augmented Generation (RAG) chain.

    Args:
    vectorstore: The vector store to use for retrieval.
    embd: The embedding model to use for encoding queries.
    model: The language model to use for generation.

    Returns:
    A callable RAG chain that takes a question and returns an answer.
    """
    if VERBOSE:
        print("Creating RAG chain")

    # Load the RAG prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Create a retriever function
    def retriever(query):
        results = similarity_search(query, vectorstore, embd)
        return [Document(page_content=result['text'], metadata=result['metadata']) for result in results]

    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain

def query_rag_chain(rag_chain, question: str) -> str:
    """
    Query the RAG chain with a question.

    Args:
    rag_chain: The RAG chain to query.
    question (str): The question to ask.

    Returns:
    str: The generated answer.
    """
    if VERBOSE:
        print(f"Querying RAG chain with question: {question}")

    answer = rag_chain.invoke(question)
    return answer
