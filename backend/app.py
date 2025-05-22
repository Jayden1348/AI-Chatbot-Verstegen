from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

app = Flask(__name__,  
    static_folder="../frontend/static",  
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
        response = ollama.chat(
            model="openchat",  # or the name of the model you pulled
            messages=[
                {"role": "system", "content": "Beantwoord alleen vragen over de CAO van Verstegen. "
                "Vakantie, contracten, loonstroken en jaaropgaven zijn te vinden in de HRToday app. "},
                {"role": "user", "content": prompt}
            ]
        )
        return jsonify({"answer": response['message']['content'].strip()})
    except Exception as e:
        return jsonify({"answer": f"Er ging iets mis: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
