from fastapi import FastAPI
from pydantic import BaseModel
import main

app = FastAPI(title="UTD Campus Assistant API")
vector_store = main.get_vector_store()
rag_chain = main.create_rag_chain(vector_store)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    print(f"Received question: {request.question}")
    print(request)
    answer = rag_chain.invoke(request.question)
    print(f"Generated answer: {answer}")
    return {"answer": answer}