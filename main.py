import vertexai
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
import pptx
import gradio as gr
from langchain.llms import VertexAI
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from vertexai.language_models import TextGenerationModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ID=os.getenv("PROJECT_ID")
LOCATION=os.getenv("LOCATION")
BUCKET=os.getenv("BUCKET")

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET,
)

# embeddings generation model
vertex_embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")


def pretty_print_docs(docs):
    """Displays loaded documents in a structured format"""
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def generate_embeddings_and_vector(texts):
    """
    Creates text embeddings using Vertex AI and builds a Chroma vector index

    Args:
        texts: A list of text chunks

    Returns:
        A Chroma vector index ready for similarity search
    """
    vector_index = Chroma.from_texts(texts, vertex_embeddings).as_retriever()
    return vector_index


def get_similar_documents(vector_index, search_query):
    """
    Finds documents semantically relevant to a query using the vector index

    Args:
        vector_index: The Chroma vector index to search within
        search_query: The user's search query

    Returns:
        A list of relevant documents
    """
    docs = vector_index.get_relevant_documents(search_query)
    return docs


def generate_final_response(docs, search_query):
    """
    Generates a concise and informative answer to the user's query, leveraging the provided documents for context.

    Args:
        docs: A list of relevant documents (likely as LangChain Document objects).
        search_query: The user's search query.

    Returns:
        A text string containing the generated answer.
    """

    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0.9,
        "top_p": 1
    }

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer: """.format(context=docs, question=search_query)

    model = TextGenerationModel.from_pretrained("text-bison")
    response = model.predict(prompt_template, **parameters)

    print(response.text)

    return response.text


def process_file(fileobj, search_query):
    """
    Loads a supported document, extracts its text content, and generates an answer to a provided query based on the document.

    Args:
        fileobj: A file-like object representing the document.
        search_query: The user's question about the document.

    Returns:
        A text string containing the answer, or "Failed to load the document" if an error occurs.
    """

    file_path = fileobj.name
    filename, file_extension = os.path.splitext(file_path)

    if file_extension == '.txt':
        # return do_something(file_path)
        loader = TextLoader(file_path)
        documents = loader.load()

        # split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in documents)
        texts = text_splitter.split_text(context)

    if file_extension == '.pdf':
        # return do_something(file_path)
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in documents)
        texts = text_splitter.split_text(context)

    if file_extension == '.pptx' or file_extension == '.ppt':
        # return do_something(file_path)
        loader = UnstructuredPowerPointLoader(file_path)
        documents = loader.load_and_split()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in documents)
        texts = text_splitter.split_text(context)

    if file_extension == '.docx' or file_extension == '.doc':
        # return do_something(file_path)
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load_and_split()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        context = "\n\n".join(str(p.page_content) for p in documents)
        texts = text_splitter.split_text(context)

    if file_extension == '.csv':
        # return do_something(file_path)
        loader = CSVLoader(file_path)
        documents = loader.load()

        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = [str(p.page_content) for p in documents]
        # texts = text_splitter.split_text(context)

    if len(texts) > 0:

        vector_index = generate_embeddings_and_vector(texts)

        llm = VertexAI(model_name="gemini-pro")
        _filter = LLMChainFilter.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=_filter, base_retriever=vector_index
        )


        compressed_docs = compression_retriever.get_relevant_documents(
            search_query
        )
        context_text = [i.page_content for i in compressed_docs]
        response_text = generate_final_response(context_text, search_query)
        # print(compressed_docs)
        pretty_print_docs(compressed_docs)
        # return docs[0].page_content
        return response_text

    else:
        return "Failed to load the document"
    
    
    
    
    
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Text Embeddings + ChromaDB + Text Bison"):

            app = gr.Interface(
                fn=process_file,
                inputs=["file", "text"],
                outputs=["textbox"],
                title="Question Answering bot",
                description="Input context and question, then get answers!",
            )

demo.launch(debug=True)
     