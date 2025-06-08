from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import csv, io

app = Flask(__name__)
app.secret_key = "super-secret-key"
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faq_multi.db'
db = SQLAlchemy(app)

# === Models ===
class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default="user")

class ChatbotResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(300), nullable=False)
    answer = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(100), nullable=True)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100))
    user_message = db.Column(db.String(300), nullable=False)
    bot_response = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'))

# === TF-IDF Cache ===
tfidf_cache = {}

def cache_tfidf(client_id):
    responses = ChatbotResponse.query.filter_by(client_id=client_id).all()
    questions = [r.question for r in responses]
    answers = [r.answer for r in responses]
    if not questions:
        return
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    tfidf_cache[client_id] = {
        "questions": questions,
        "answers": answers,
        "vectorizer": vectorizer,
        "vectors": vectors
    }

def find_closest_question(user_input, client_id, threshold=0.3):
    if client_id not in tfidf_cache:
        cache_tfidf(client_id)
    cached = tfidf_cache.get(client_id)
    if not cached:
        return "Maaf, belum ada data FAQ untuk Anda."

    input_vec = cached["vectorizer"].transform([user_input])
    similarity = cosine_similarity(input_vec, cached["vectors"])
    max_sim = similarity.max()

    if max_sim < threshold:
        return "Maaf, saya belum mengerti apa yang di maksud."

    idx = similarity.argmax()
    return cached['answers'][idx]

# === Routes ===
@app.route("/")
def chat_ui():
    return render_template("user_chat.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        client = Client.query.filter_by(username=username).first()
        if client and check_password_hash(client.password, password):
            session["client_id"] = client.id
            session["username"] = client.username
            session["role"] = client.role
            flash("Login berhasil!", "success")
            return redirect(url_for("index"))
        flash("Username atau password salah.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/index")
def index():
    if "client_id" not in session:
        return redirect(url_for("login"))
    selected_category = request.args.get("category")
    page = int(request.args.get("page", 1))
    per_page = 5
    query = ChatbotResponse.query.filter_by(client_id=session["client_id"])
    if selected_category:
        query = query.filter_by(category=selected_category)
    total = query.count()
    faqs = query.offset((page - 1) * per_page).limit(per_page).all()
    categories = [row[0] for row in db.session.query(ChatbotResponse.category).filter_by(client_id=session["client_id"]).distinct()]
    return render_template("index.html", faqs=faqs, page=page, total_pages=(total + per_page - 1) // per_page,
                           selected_category=selected_category, categories=categories)

@app.route("/chat", methods=["POST"])
def chat():
    if "client_id" not in session:
        return jsonify({"response": "Silakan login dulu."})
    data = request.get_json()
    user_message = data.get("message")
    session_id = data.get("session_id")
    bot_response = find_closest_question(user_message, session["client_id"])
    db.session.add(ChatHistory(
        user_message=user_message,
        bot_response=bot_response,
        session_id=session_id,
        client_id=session["client_id"]
    ))
    db.session.commit()
    return jsonify({"response": bot_response})


@app.route("/add", methods=["POST"])
def add_faq():
    if "client_id" not in session:
        return redirect(url_for("login"))
    q, a, cat = request.form["new_question"], request.form["new_answer"], request.form.get("category")
    if not ChatbotResponse.query.filter_by(question=q, client_id=session["client_id"]).first():
        db.session.add(ChatbotResponse(question=q, answer=a, category=cat, client_id=session["client_id"]))
        db.session.commit()
        cache_tfidf(session["client_id"])
        flash("FAQ berhasil ditambahkan.", "success")
    else:
        flash("Pertanyaan sudah ada!", "warning")
    return redirect(url_for("index"))

@app.route("/upload", methods=["POST"])
def upload():
    if "client_id" not in session:
        return redirect(url_for("login"))
    file = request.files["csv_file"]
    if file and file.filename.endswith(".csv"):
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        reader = csv.reader(stream)
        added, skipped = 0, 0
        for row in reader:
            if len(row) >= 2:
                q, a = row[0], row[1]
                cat = row[2] if len(row) > 2 else None
                if not ChatbotResponse.query.filter_by(question=q, client_id=session["client_id"]).first():
                    db.session.add(ChatbotResponse(question=q, answer=a, category=cat, client_id=session["client_id"]))
                    added += 1
                else:
                    skipped += 1
        db.session.commit()
        cache_tfidf(session["client_id"])
        flash(f"{added} ditambahkan. {skipped} dilewati (duplikat).", "success")
    else:
        flash("Format file tidak valid. Harus CSV.", "danger")
    return redirect(url_for("index"))

@app.route("/edit/<int:id>", methods=["POST"])
def edit_faq(id):
    faq = ChatbotResponse.query.get_or_404(id)
    if faq.client_id != session.get("client_id"):
        flash("Tidak diizinkan.", "danger")
        return redirect(url_for("index"))
    faq.question = request.form['question']
    faq.answer = request.form['answer']
    faq.category = request.form.get('category')
    db.session.commit()
    cache_tfidf(session["client_id"])
    flash("FAQ berhasil diperbarui.", "success")
    return redirect(url_for("index"))

@app.route("/delete/<int:id>")
def delete_faq(id):
    faq = ChatbotResponse.query.get_or_404(id)
    if faq.client_id != session.get("client_id"):
        flash("Tidak diizinkan.", "danger")
        return redirect(url_for("index"))
    db.session.delete(faq)
    db.session.commit()
    cache_tfidf(session["client_id"])
    flash("FAQ dihapus.", "success")
    return redirect(url_for("index"))

@app.route("/bulk_delete", methods=["POST"])
def bulk_delete():
    if "client_id" not in session:
        return redirect(url_for("login"))
    ids = request.form.getlist("selected_ids")
    if ids:
        ChatbotResponse.query.filter(ChatbotResponse.id.in_(ids), ChatbotResponse.client_id == session["client_id"]).delete(synchronize_session=False)
        db.session.commit()
        cache_tfidf(session["client_id"])
    return redirect(url_for("index"))

@app.route("/history")
def history():
    if "client_id" not in session:
        return redirect(url_for("login"))
    chats = ChatHistory.query.filter_by(client_id=session["client_id"]).order_by(ChatHistory.timestamp.desc()).all()
    return render_template("history.html", chats=chats)

@app.route("/clear-history")
def clear_history():
    if "client_id" not in session:
        return redirect(url_for("login"))
    ChatHistory.query.filter_by(client_id=session["client_id"]).delete()
    db.session.commit()
    flash("Riwayat berhasil dihapus.", "success")
    return redirect(url_for("history"))

@app.route("/healthz")
def healthz():
    return "OK"

# === Setup DB dan admin ===
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not Client.query.filter_by(username="admin").first():
            admin = Client(username="admin", password=generate_password_hash("admin123"), role="admin")
            db.session.add(admin)
            db.session.commit()
            print("Admin default dibuat.")
    app.run(debug=True)
