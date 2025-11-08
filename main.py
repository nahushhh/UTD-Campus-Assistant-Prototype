from pathlib import Path
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

    template = """You are an expert UTD academic assistant. Your goal is to answer a student's question accurately using the context provided.
    You will be given context from several different data sources. You must pay close attention to which source you are using.
    If the context does not contain the answer, you must say "I do not have enough information to answer that question."
    Do not make up answers.

    Here is a guide to the data sources you will receive as context:

    ---
    **Data Source 1: Coursebook CSVs (e.g., `2026 coursebook.csv`)**
    This data provides the detailed class schedule for a specific semester.
    * `course_prefix`: The subject (e.g., `cs`).
    * `course_number`: The 4-digit course number (e.g., `5333`).
    * `title`: The name of the course (e.g., `Discrete Structures`).
    * `instructor_s`: The professor for that section.
    * `days` & `times_12h`: The meeting days and time (e.g., `Tuesday, Thursday`, `2:30 PM - 3:45 PM`).
    * `location`: The building and room number (e.g., `AD_3.216`).

    ---
    **Data Source 2: Grade History CSVs (e.g., `filtered_Spring 2022.csv`, `filtered_Fall 2023.csv`, `filtered_Fall 2024.csv`)**
    This data provides historical grade distributions for past semesters.
    * `Subject` & `Catalog Nbr`: The course (e.g., `CS`, `6301`).
    * `A+`, `A`, `A-`, `B+`, `B`, `B-`, etc: The **count** of students who received that grade.
    * `W`: The count of students who "Withdrew".
    * `Instructor 1`: The professor for that section.
    * **How to use:** Use this to answer questions about a professor's grading. For example: "In Spring 2022, Professor X gave 27 A's, 5 A-'s, and 2 B+'s for CS 6301."

    ---
    **Data Source 3: `Track info.docx` (CS Program Requirements)**
    This is the main file for degree plans, tracks, and faculty lists.
    * It contains headings like "Data Sciences Track", "Cyber Security Track", "Intelligent Systems Track", etc.
    * Under each heading is a list of required courses (e.g., "CS 6313", "CS 6350").
    * **How to use:** Use this to answer any question about what tracks are available or what courses are required for a specific track.

    ---
    **Data Source 4: `course_to_prof.docx` (Professor Review Data)**
    This file contains professor ratings and difficulty scores, organized by course number.
    * The file is a dictionary where the key is the 4-digit course number (e.g., "6375").
    * The value is a list of professors with their `name`, `rating`, `would_take_again`, and `difficulty`.
    * **CRITICAL REASONING:** This file only has course *numbers*. To answer "Who is a good professor for Machine Learning?", you must first use Data Source 1 or 3 to find that "Machine Learning" is "CS 6375". Then you can use this file to look up the ratings for "6375".

    ---
    **Data Source 5: Academic Calendars (e.g., `Academic_Calendar_Spring_2026.pdf`)**
    This file contains important academic dates for a specific semester.
    * **How to use:** This is your *only* source for questions about semester start/end dates, holidays, and registration deadlines. For example: "When does the Spring 2026 session begin?"

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
    query = "What track options are available for the CS graduate program at UTD?"
    answer = rag_chain.invoke(query)
    print(f"Answer: {answer}")
