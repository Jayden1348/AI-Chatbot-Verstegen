from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import chromadb
import json
import requests

app = Flask(__name__,   static_folder="../frontend/static",  
    template_folder="../frontend"  
    )

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="my_documents")

with open("../Data/Verstegen_Cao.txt", "r", encoding="utf-8") as f:
    text_chunks = f.read().split(". ")

if not collection.count():
    for i, chunk in enumerate(text_chunks):
        collection.add(documents=[chunk], embeddings=[model.encode(chunk)], ids=[str(i)])

@app.route("/")
def index():
 return send_from_directory(app.template_folder, "frontend.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    query_embedding = model.encode(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n".join(results['documents'][0])
    prompt = f"Context:\n{context}\n\nVraag: {question}\nAntwoord in het Nederlands:"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "openchat", "prompt": prompt},
            stream=True
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode("utf-8"))
                    full_response += json_data.get("response", "")
                except json.JSONDecodeError:
                    continue

        return jsonify({"answer": full_response.strip()})
    except Exception as e:
        return jsonify({"answer": f"Er ging iets mis: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
