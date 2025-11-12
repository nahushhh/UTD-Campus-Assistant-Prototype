from pathlib import Path
import json
import time
import csv
import boto3
import zipfile
import shutil
import os
from botocore.config import Config
from langchain_core.documents import Document
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

FAISS_VECTOR_STORE_PATH = "faiss_vector_store"
S3_BUCKET_NAME = "utdca-vector-db"
S3_INDEX_NAME = "faiss_vector_store.zip"

retry_config = Config(
    retries = {
        'max_attempts': 10,
        'mode': 'adaptive'
    }
)

s3_client = boto3.client("s3", config=retry_config)

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    config = retry_config
)

llm = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",

)

embedding_model = BedrockEmbeddings(
    client=bedrock_client,
    model_id="cohere.embed-v4:0",
)

# --- NEW: Helper function to break a list into batches ---
def batch_generator(data, batch_size):
    """Yields batches of a specific size from a list."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def _load_prof_reviews_json(file_path):
    """
    Reads the professor review JSON file and transforms each entry
    into a semantically rich Document.
    """
    docs = []
    print(f"Starting to load {file_path.name} with custom JSON logic...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # data is a dict: {"6313": [...], "6350": [...]}
        for course_number, professors_list in data.items():
            for prof in professors_list:
                # Transform into a clean sentence
                page_content = (
                    f"For course CS {course_number}, professor {prof.get('name')} "
                    f"has a student rating of {prof.get('rating')}, "
                    f"a difficulty of {prof.get('difficulty')}, "
                    f"and {prof.get('would_take_again')} of students would take them again."
                )
                metadata = {
                    "source": "professor_reviews", # Generic, non-revealing source name
                    "course": course_number,
                    "professor": prof.get('name')
                }
                docs.append(Document(page_content=page_content, metadata=metadata))
    print(f"Finished loading {len(docs)} professor reviews from {file_path.name}.")
    return docs

def _load_grade_history_csv(file_path):
    """
    Reads a grade history CSV and transforms each row into a
    semantically rich Document.
    """
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Clean up keys (some have spaces)
            row = {k.strip(): v for k, v in row.items()}
            
            # Combine all grade counts into a single string
            grades = [f"{k}: {v}" for k, v in row.items() if k in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F', 'W', 'P'] and v]
            grade_str = ", ".join(grades)
            
            # Create a natural language sentence
            page_content = (
                f"In a past semester, for course {row['Subject']} {row['Catalog Nbr']} Section {row['Section']}, "
                f"the instructor {row['Instructor 1']} (and others: {row.get('Instructor 2', 'N/A')}, {row.get('Instructor 3', 'N/A')}) "
                f"gave the following grades: {grade_str}."
            )
            metadata = {
                "source": str(file_path.name), 
                "row": i, 
                "course": f"{row['Subject']} {row['Catalog Nbr']}", 
                "professor": row['Instructor 1']
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs

# --- NEW: Custom function to load and transform coursebook CSVs ---
def _load_coursebook_csv(file_path):
    """
    Reads a coursebook CSV and transforms each row into a
    semantically rich Document.
    """
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            page_content = (
                f"Course Section: {row['course_prefix']} {row['course_number']}.{row['section']} (Class Number: {row['class_number']}). "
                f"Title: {row['title']}. "
                f"Instructor: {row['instructor_s']}. "
                f"Schedule: {row['days']} from {row['times_12h']}. "
                f"Location: {row['location']}. "
                f"Status: {row['enrolled_status']} ({row['enrolled_current']}/{row['enrolled_max']} enrolled)."
            )
            metadata = {
                "source": str(file_path.name), 
                "row": i, 
                "course": f"{row['course_prefix']} {row['course_number']}"
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def get_vector_store():
    index_path = Path(FAISS_VECTOR_STORE_PATH)
    try:
        if index_path.exists():
            print("Loading existing vector store...")
            vector_store = FAISS.load_local(FAISS_VECTOR_STORE_PATH, 
                                            embedding_model, 
                                            allow_dangerous_deserialization=True
                                            )
            print("Vector store loaded.")
            return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")

    print("No local index found. Checking S3...")

    try:
        s3_client.download_file(S3_BUCKET_NAME, S3_INDEX_NAME, S3_INDEX_NAME)
        print(f"Successfully downloaded {S3_INDEX_NAME} from S3.")

        print(f"Unzipping {S3_INDEX_NAME} to {FAISS_VECTOR_STORE_PATH}...")
        with zipfile.ZipFile(S3_INDEX_NAME, 'r') as zip_ref:
            zip_ref.extractall(FAISS_VECTOR_STORE_PATH)
        print("Unzipping completed.")

        os.remove(S3_INDEX_NAME)

        print(f"Loading vector store from unzipped folder: {FAISS_VECTOR_STORE_PATH}")
        return FAISS.load_local(
            FAISS_VECTOR_STORE_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Failed to download or load index from S3: {e}. Building from scratch...")

    print("Creating a new vector store...")
    data_dir = Path("data")
    all_docs = []
    print("Loading documetns from data folder...")
    for file_path in data_dir.glob("*"):
        try:
            if file_path.suffix == '.pdf':
                loader = PyPDFLoader(str(file_path))
                print(f"-> Loading PDF: {file_path.name}")
                all_docs.extend(loader.load())
            elif file_path.suffix == '.docx':
                loader = Docx2txtLoader(str(file_path))
                print(f"-> Loading DOCX: {file_path.name}")
                all_docs.extend(loader.load())
            elif file_path.suffix == '.csv':
                if "coursebook" in file_path.name.lower():
                    print(f"-> Loading Coursebook CSV: {file_path.name}")
                    all_docs.extend(_load_coursebook_csv(file_path))
                elif "filtered_" in file_path.name.lower():
                    print(f"-> Loading Grade History CSV: {file_path.name}")
                    all_docs.extend(_load_grade_history_csv(file_path))
                else:
                    print(f"-> Skipping unknown CSV: {file_path.name}")
            else:
                print(f"-> Skipping unsupported file: {file_path.name}")
                continue
        except Exception as e:
            print(f"Error loading documents: {e}")
    
    if not all_docs:
        print("No documents were loaded. Please add files to the 'data' folder.")
        return None

    print(f"\nLoaded a total of {len(all_docs)} document sections.")

    # Splitting the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )
    docs = text_splitter.split_documents(all_docs)
    print(f"Split into {len(docs)} chunks.")

    # Creating vector store
    print("Creating vector store with batching to avoid throttling...")
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    batch_size = 100
    text_batches = list(batch_generator(texts, batch_size))
    metadata_batches = list(batch_generator(metadatas, batch_size))

    vector_store = None
    total_batches = len(text_batches)

    for i, (text_batch, metadata_batch) in enumerate(zip(text_batches, metadata_batches)):
        print(f"Processing batch {i + 1}/{total_batches}...")
        if i == 0:
            # Create the store with the first batch
            vector_store = FAISS.from_texts(
                text_batch, embedding_model, metadatas=metadata_batch
            )
        else:
            # Add subsequent batches to the existing store
            vector_store.add_texts(
                text_batch, metadatas=metadata_batch
            )
        
        print(f"Batch {i + 1} processed. Sleeping for 1 second...")
        time.sleep(3)  # Pause for 1 second to respect rate limits
    
    if vector_store:
        print("Vector store created successfully.")
        vector_store.save_local(FAISS_VECTOR_STORE_PATH)
        print("Vector store saved locally.")
    else:
        print("Vector store could not be created (e.g., no documents found).")

    try:
        print(f"Zipping index folder to {S3_INDEX_NAME}...")
        shutil.make_archive(FAISS_VECTOR_STORE_PATH, 'zip', FAISS_VECTOR_STORE_PATH)
        print("Upload to S3...")
        s3_client.upload_file(S3_INDEX_NAME, S3_BUCKET_NAME, S3_INDEX_NAME)
        print("Upload completed.")
        os.remove(S3_INDEX_NAME)
    except Exception as e:
        print(f"Failed to upload index to S3: {e}")
    
    return vector_store
   
def create_rag_chain(vector_store):
    print("Creating RAG chain...")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":10})

    template = """You are an expert UTD academic assistant. Your goal is to answer a student's question accurately, concisely, and directly.
    
    Use the provided context to find the answer. Synthesize the information from all relevant context pieces to form a complete answer.
    
    **CRITICAL RULES:**
    1.  **NEVER** mention the "context", "provided documents", or "information provided". Act as if you know this information innately.
    2.  **NEVER** say "Based on the context..." or "According to the document...".
    3.  If the context does not contain the answer, simply state: "I do not have that information."
    4.  Do not make up any information that is not in the context.
    5.  Be direct. If the user asks for a list, provide a list. If they ask a yes/no question, answer it directly.
    6.  DO NOT MENTION OR REFERENCE ANY DATA SOURCES OR METADATA FROM THE CONTEXT.
    7.  Keep your answers brief and to the point. Be natural and human like. Be polite and professional.
    8.  DO NOT share any internal information about the data sources, data content, UTD systems, processes, or data handling.

    ---
    HERE IS THE CONTEXT:
    {context}
    
    HERE IS THE QUESTION:
    Question: {question}
    
    YOUR ANSWER:
    """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__=="__main__":
    vector_store = get_vector_store()
    rag_chain = create_rag_chain(vector_store)
    query = "What are the meeting times and locations for 'Discrete Structures' in Spring 2026?"
    answer = rag_chain.invoke(query)
    print(f"Answer: {answer}")
