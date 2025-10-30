import boto3
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain.prompts import PromptTemplate

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

llm = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",

)

embedding_model = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1",
)

def create_vector_store():
    # Loading file
    loader = PyPDFLoader("Academic_Calendar_Fall_2025.pdf")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # Splitting the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # Creating vector store
    print("Creating vector store...")
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vector_store = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    print("Vector store created.")

    return vector_store

def create_rag_chain(vector_store):
    print("Creating RAG chain...")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

    template = """ You are an academic assistant. Use the following context to answer the question at the end.
    {context}
    Question: {question}
    If you don't know the answer, just say that you don't know. Do not try to make up an answer.""" 

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__=="__main__":
    vector_store = create_vector_store()
    rag_chain = create_rag_chain(vector_store)
    query = "When does the fall semester start in 2025?"
    answer = rag_chain.invoke(query)
    print(f"Answer: {answer}")
