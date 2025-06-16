from flask import Flask, request, jsonify, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import bcrypt
import jwt
import chromadb
import ollama
import subprocess
import nltk
import json
from uuid import uuid4
import pygame
import time
import os


SECRET_KEY = "your-secret-key"
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRATION_MINUTES = 30

app = Flask(__name__,
            static_folder="../frontend/static",
            template_folder="../frontend")
CORS(app)
app.secret_key = "anonymous-session-key"  # Voor sessies zonder login

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.LargeBinary, nullable=False)


model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient()
collection = client.get_or_create_collection(name="my_documents")


def get_reference_info():
    with open("../Data/Verstegen_Cao.txt", "r", encoding="utf-8") as f:

        text_chunks = nltk.sent_tokenize(f.read())

    if collection.count() == 0:
        embeddings = model.encode(text_chunks, convert_to_numpy=True)
        collection.add(documents=text_chunks, embeddings=embeddings, ids=[
                       str(i) for i in range(len(text_chunks))])


def search_documents(question):
    query_embedding = model.encode(question)
    results = collection.query(query_embeddings=[query_embedding])

    if 'documents' in results and results['documents']:
        return "\n".join(results['documents'][0])
    return "No relevant documents found."


def create_tables():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="HR1").first():
            hashed_pw = bcrypt.hashpw(
                "Kruidje".encode("utf-8"), bcrypt.gensalt())
            user = User(username="HR1", password=hashed_pw)
            db.session.add(user)
            db.session.commit()
            print("Gebruiker 'HR1' aangemaakt.")
        else:
            print("Gebruiker 'HR1' bestaat al.")


def create_token(username):
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise jwt.ExpiredSignatureError("Token is verlopen.")
    except jwt.InvalidTokenError:
        raise jwt.InvalidTokenError("Ongeldige token.")


def speak_dutch(text):
    tts = gTTS(text=text, lang='nl')
    filename = "dutch_speech.mp3"
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()
    os.remove(filename)


@app.post("/api/login")
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password):
        return jsonify({"error": "Ongeldige gebruikersnaam of wachtwoord."}), 401

    token = create_token(username)
    return jsonify({"access_token": token, "token_type": "bearer"})


@app.post("/register")
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Gebruikersnaam en wachtwoord zijn verplicht."}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Gebruiker bestaat al."}), 400

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    new_user = User(username=username, password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "Registratie gelukt."}), 201


@app.get("/me")
def get_current_user():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Geen of ongeldige Authorization-header."}), 401

    token = auth_header.split(" ")[1]
    try:
        username = decode_token(token)
        return jsonify({"username": username})
    except Exception as e:
        return jsonify({"error": str(e)}), 401


@app.route("/")
def index():
    return send_from_directory(app.template_folder, "frontend.html")


@app.route("/login")
def login_page():
    return send_from_directory(app.template_folder, "login.html")


@app.route("/dashboard")
def dashboard():
    return send_from_directory(app.template_folder, "dashboard.html")


@app.route("/speak_dutch", methods=["POST"])
def speak_dutch_api():
    data = request.json
    text = data.get("text")
    if text:
        speak_dutch(text)
        return jsonify({"status": "ok"})
    return jsonify({"error": "No text provided"}), 400


@app.route("/ask", methods=["POST"])
def ask():
    if "chat_id" not in session:
        session["chat_id"] = str(uuid4())[:8]

    data = request.json
    question = data.get("question", "")
    context = search_documents(question)

    prompt = f"Context:\n{context}\n\nVraag: {question}\nAntwoord in het Nederlands:"

    try:
        response = ollama.chat(
            model="openchat",
            messages=[
                {"role": "system", "content": "Beantwoord alleen vragen met antwoorden die te halen zijn uit de gegeven context van Verstegen. "
                 "Beantwoord deze beknopt. Als er geen geschikt antwoord kan worden gevonden, verwijs dan naar het telefoonnummer (010) 245 51 00. "
                 "Vakantie, contracten, loonstroken en jaaropgaven zijn te vinden in het werknemersportaal."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response['message']['content'].strip()

        log_entry = {
            "user": question,
            "bot": answer
        }

        os.makedirs("chatlogs", exist_ok=True)
        filepath = "chatlogs/sessions.json"

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                all_logs = json.load(f)
        else:
            all_logs = {}

        chat_id = session["chat_id"]
        if chat_id not in all_logs:
            all_logs[chat_id] = []

        all_logs[chat_id].append(log_entry)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_logs, f, ensure_ascii=False, indent=2)

        return jsonify({"answer": answer, "chat_id": chat_id})
    except Exception as e:
        print(f"UNEXPECTED ERROR IN ASK FUNCTION: {str(e)}")
        return jsonify({"answer": "Er ging iets mis."}), 500


@app.get("/history")
def get_chat_history():
    chat_id = session.get("chat_id")
    if not chat_id:
        return jsonify({"history": []})

    filepath = "chatlogs/sessions.json"
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            all_logs = json.load(f)
            return jsonify({"history": all_logs.get(chat_id, [])})
    return jsonify({"history": []})


@app.get("/all_chats")
def get_all_chats():
    filepath = "chatlogs/sessions.json"
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            all_logs = json.load(f)
            return jsonify(all_logs)
    return jsonify({})


if __name__ == "__main__":
    print("Natural Language Toolkit wordt bijgewerkt...")
    nltk.download('punkt_tab')
    print("Ollama wordt gestart...")
    process = subprocess.Popen(
        ['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Database tables worden geinitialiseerd...")
    create_tables()
    print("Model wordt geinitialiseerd...")
    get_reference_info()
    print("Web applicatie wordt gestart...")
    app.run(debug=True)

