import torch
from src.embeddings import embed_texts
from src.utils import chunk_text

def retrieve(query, tokenizer, model, device, faiss_index, documents, top_k=3):
    query_emb = embed_texts([query], tokenizer, model, device)
    distances, indices = faiss_index.search(query_emb, top_k)
    retrieved_chunks = [documents[i] for i in indices[0]]
    return retrieved_chunks

def generate_response(query, tokenizer, model, device, faiss_index, documents, top_k=3):
    retrieved_chunks = retrieve(query, tokenizer, model, device, faiss_index, documents, top_k)
    context = " ".join(retrieved_chunks)
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=150, temperature=0.2)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer
