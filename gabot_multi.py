
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import csv, io, os
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super-secret-key-change-in-production")
CORS(app)

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///faq_multi.db")
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

db = SQLAlchemy(app)

# === Models ===
class Client(db.Model):
    __tablename__ = 'clients'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default="user")

class ChatbotResponse(db.Model):
    __tablename__ = 'chatbot_responses'
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(300), nullable=False, index=True)
    answer = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(100), nullable=True, index=True)
    client_id = db.Column(db.Integer, db.ForeignKey('clients.id'), nullable=False, index=True)

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), index=True)
    user_message = db.Column(db.String(300), nullable=False)
    bot_response = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    client_id = db.Column(db.Integer, db.ForeignKey('clients.id'), index=True)

# === Rate Limiting ===
from collections import defaultdict, deque
import time

request_counts = defaultdict(lambda: deque())
RATE_LIMIT_PER_MINUTE = 60

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        while request_counts[client_ip] and request_counts[client_ip][0] < minute_ago:
            request_counts[client_ip].popleft()
        
        if len(request_counts[client_ip]) >= RATE_LIMIT_PER_MINUTE:
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        request_counts[client_ip].append(now)
        return f(*args, **kwargs)
    return decorated_function

# === TF-IDF Cache with error handling ===
tfidf_cache = {}

def cache_tfidf(client_id):
    try:
        responses = ChatbotResponse.query.filter_by(client_id=client_id).all()
        questions = [r.question for r in responses]
        answers = [r.answer for r in responses]
        if not questions:
            return
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        vectors = vectorizer.fit_transform(questions)
        tfidf_cache[client_id] = {
            "questions": questions,
            "answers": answers,
            "vectorizer": vectorizer,
            "vectors": vectors
        }
        logger.info(f"TF-IDF cache updated for client {client_id}")
    except Exception as e:
        logger.error(f"Error caching TF-IDF for client {client_id}: {str(e)}")

def find_closest_question(user_input, client_id, threshold=0.3):
    try:
        if client_id not in tfidf_cache:
            cache_tfidf(client_id)
        cached = tfidf_cache.get(client_id)
        if not cached:
            return "Maaf, belum ada data FAQ untuk Anda."

        input_vec = cached["vectorizer"].transform([user_input])
        similarity = cosine_similarity(input_vec, cached["vectors"])
        max_sim = similarity.max()

        if max_sim < threshold:
            return "Maaf, saya belum mengerti apa yang dimaksud. Silakan coba pertanyaan lain atau hubungi admin."

        idx = similarity.argmax()
        return cached['answers'][idx]
    except Exception as e:
        logger.error(f"Error finding closest question: {str(e)}")
        return "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda."

# === Routes ===
@app.route("/")
def chat_ui():
    return render_template("user_chat.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
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
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash("Terjadi kesalahan sistem.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/index")
def index():
    if "client_id" not in session:
        return redirect(url_for("login"))
    try:
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
    except Exception as e:
        logger.error(f"Index error: {str(e)}")
        flash("Terjadi kesalahan dalam memuat data.", "danger")
        return render_template("index.html", faqs=[], page=1, total_pages=0, categories=[])

@app.route("/chat", methods=["POST"])
@rate_limit
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"response": "Pesan tidak valid."}), 400
            
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Pesan kosong."}), 400
            
        session_id = data.get("session_id")
        
        # Default client for public access
        client_id = session.get("client_id", 1)  # Default to admin for public chat
        
        bot_response = find_closest_question(user_message, client_id)
        
        # Save chat history
        chat_record = ChatHistory(
            user_message=user_message,
            bot_response=bot_response,
            session_id=session_id,
            client_id=client_id
        )
        db.session.add(chat_record)
        db.session.commit()
        
        return jsonify({"response": bot_response})
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"response": "Maaf, terjadi kesalahan sistem."}), 500

@app.route("/add", methods=["POST"])
def add_faq():
    if "client_id" not in session:
        return redirect(url_for("login"))
    try:
        q, a, cat = request.form["new_question"], request.form["new_answer"], request.form.get("category")
        if not ChatbotResponse.query.filter_by(question=q, client_id=session["client_id"]).first():
            db.session.add(ChatbotResponse(question=q, answer=a, category=cat, client_id=session["client_id"]))
            db.session.commit()
            cache_tfidf(session["client_id"])
            flash("FAQ berhasil ditambahkan.", "success")
        else:
            flash("Pertanyaan sudah ada!", "warning")
    except Exception as e:
        logger.error(f"Add FAQ error: {str(e)}")
        flash("Terjadi kesalahan saat menambah FAQ.", "danger")
    return redirect(url_for("index"))

@app.route("/upload", methods=["POST"])
def upload():
    if "client_id" not in session:
        return redirect(url_for("login"))
    try:
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
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        flash("Terjadi kesalahan saat upload file.", "danger")
    return redirect(url_for("index"))

@app.route("/edit/<int:id>", methods=["POST"])
def edit_faq(id):
    try:
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
    except Exception as e:
        logger.error(f"Edit FAQ error: {str(e)}")
        flash("Terjadi kesalahan saat edit FAQ.", "danger")
    return redirect(url_for("index"))

@app.route("/delete/<int:id>")
def delete_faq(id):
    try:
        faq = ChatbotResponse.query.get_or_404(id)
        if faq.client_id != session.get("client_id"):
            flash("Tidak diizinkan.", "danger")
            return redirect(url_for("index"))
        db.session.delete(faq)
        db.session.commit()
        cache_tfidf(session["client_id"])
        flash("FAQ dihapus.", "success")
    except Exception as e:
        logger.error(f"Delete FAQ error: {str(e)}")
        flash("Terjadi kesalahan saat hapus FAQ.", "danger")
    return redirect(url_for("index"))

@app.route("/bulk_delete", methods=["POST"])
def bulk_delete():
    if "client_id" not in session:
        return redirect(url_for("login"))
    try:
        ids = request.form.getlist("selected_ids")
        if ids:
            ChatbotResponse.query.filter(ChatbotResponse.id.in_(ids), ChatbotResponse.client_id == session["client_id"]).delete(synchronize_session=False)
            db.session.commit()
            cache_tfidf(session["client_id"])
        flash("FAQ terpilih berhasil dihapus.", "success")
    except Exception as e:
        logger.error(f"Bulk delete error: {str(e)}")
        flash("Terjadi kesalahan saat hapus bulk FAQ.", "danger")
    return redirect(url_for("index"))

@app.route("/history")
def history():
    if "client_id" not in session:
        return redirect(url_for("login"))
    try:
        chats = ChatHistory.query.filter_by(client_id=session["client_id"]).order_by(ChatHistory.timestamp.desc()).all()
        return render_template("history.html", chats=chats)
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return render_template("history.html", chats=[])

@app.route("/clear-history")
def clear_history():
    if "client_id" not in session:
        return redirect(url_for("login"))
    try:
        ChatHistory.query.filter_by(client_id=session["client_id"]).delete()
        db.session.commit()
        flash("Riwayat berhasil dihapus.", "success")
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        flash("Terjadi kesalahan saat hapus riwayat.", "danger")
    return redirect(url_for("history"))

@app.route("/analytics")
def analytics():
    if "client_id" not in session:
        return redirect(url_for("login"))
    
    try:
        client_id = session["client_id"]
        
        # Total chats
        total_chats = ChatHistory.query.filter_by(client_id=client_id).count()
        
        # Most asked questions (from chat history)
        most_asked = db.session.query(
            ChatHistory.user_message, 
            db.func.count(ChatHistory.user_message).label('count')
        ).filter_by(client_id=client_id).group_by(
            ChatHistory.user_message
        ).order_by(db.desc('count')).limit(10).all()
        
        # Chat activity by date
        chat_activity = db.session.query(
            db.func.date(ChatHistory.timestamp).label('date'),
            db.func.count(ChatHistory.id).label('count')
        ).filter_by(client_id=client_id).group_by(
            db.func.date(ChatHistory.timestamp)
        ).order_by(db.desc('date')).limit(30).all()
        
        # Total FAQ count
        total_faqs = ChatbotResponse.query.filter_by(client_id=client_id).count()
        
        return render_template("analytics.html", 
                             total_chats=total_chats,
                             most_asked=most_asked,
                             chat_activity=chat_activity,
                             total_faqs=total_faqs)
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return render_template("analytics.html", total_chats=0, most_asked=[], chat_activity=[], total_faqs=0)

@app.route("/healthz")
def healthz():
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error_code=404, error_message="Halaman tidak ditemukan"), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    logger.error(f"Internal server error: {str(error)}")
    return render_template('error.html', error_code=500, error_message="Terjadi kesalahan server"), 500

# === Setup DB dan admin ===
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not Client.query.filter_by(username="admin").first():
            admin = Client(username="admin", password=generate_password_hash("admin123"), role="admin")
            db.session.add(admin)
            db.session.commit()
            print("Admin default dibuat.")
            
        # Cache TF-IDF for existing clients
        clients = Client.query.all()
        for client in clients:
            cache_tfidf(client.id)
            
    # Production settings
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    
    app.run(host="0.0.0.0", port=port, debug=debug)
