from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from collections import Counter
import csv, io
import enum
from sqlalchemy import Enum
from decorators import admin_required, client_required

class RoleEnum(enum.Enum):
    client = "client"
    admin = "admin"

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
    role = db.Column(Enum(RoleEnum), default=RoleEnum.client, nullable=False)
    business_name = db.Column(db.String(150), nullable=True)
    whatsapp = db.Column(db.String(20), nullable=True)
    is_approved = db.Column(db.Boolean, default=False)
    slug = db.Column(db.String(100), unique=True)

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

# --- Slug generator ---
def generate_slug(name):
    import re
    return re.sub(r'\W+', '', name.lower())
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

# --- Modifikasi di /register ---
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        business_name = request.form["business_name"]
        whatsapp = request.form["whatsapp"]

        if Client.query.filter_by(username=username).first():
            flash("Username sudah digunakan.", "warning")
            return redirect(url_for("register"))

        slug = generate_slug(business_name)
        if Client.query.filter_by(slug=slug).first():
            flash("Slug sudah digunakan.", "warning")
            return redirect(url_for("register"))

        new_client = Client(
            username=username,
            password=generate_password_hash(password),
            business_name=business_name,
            whatsapp=whatsapp,
            role=RoleEnum.client,  # ✅ role langsung dari Enum, aman
            is_approved=False,
            slug=slug
        )
        db.session.add(new_client)
        db.session.commit()
        flash("Pendaftaran berhasil! Tunggu konfirmasi dari admin.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        client = Client.query.filter_by(username=username).first()

        if client and check_password_hash(client.password, password):
            if not client.is_approved:
                flash("Akun Anda masih menunggu persetujuan admin.", "warning")
                return redirect(url_for("login"))
 
            session["client_id"] = client.id
            session["username"] = client.username
            session["role"] = client.role.value

            flash("Login berhasil!", "success")

            if client.role == RoleEnum.admin:
                return redirect(url_for("superadmin"))
            else:
                return redirect(url_for("index"))  # user biasa masuk ke halaman index
        else:
            flash("Username atau password salah.", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# === Public Chat by Slug ===
@app.route("/chat/<slug>")
def public_chat(slug):
    client = Client.query.filter_by(slug=slug).first()
    if not client or not client.is_approved:
        return "Client tidak ditemukan atau belum di-approve", 404
    session["client_id"] = client.id
    return render_template("user_chat.html")

# === QR Code Generator ===
@app.route("/qr/<slug>")
def generate_qr(slug):
    url = request.url_root + "chat/" + slug
    img = qrcode.make(url)
    buffer = BytesIO()
    img.save(buffer)
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')

@app.route("/index")
@client_required
def index():
    if "client_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "client":
        flash("Akses ditolak.", "danger")
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

    client = Client.query.get(session["client_id"])

    return render_template("index.html", 
        faqs=faqs, 
        page=page, 
        total_pages=(total + per_page - 1) // per_page,
        selected_category=selected_category, 
        categories=categories,
        client=client
    )


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
@client_required
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

@app.route("/analytics")
@client_required
def analytics():
    if "client_id" not in session:
        return redirect(url_for("login"))

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
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    chat_activity = db.session.query(
        db.func.date(ChatHistory.timestamp).label('date'),
        db.func.count(ChatHistory.id).label('count')
    ).filter(
        ChatHistory.client_id == client_id,
        ChatHistory.timestamp >= thirty_days_ago
    ).group_by(
        db.func.date(ChatHistory.timestamp)
    ).order_by(
        db.asc('date')
    ).all()
    default_unanswered_msg = "Maaf, saya belum mengerti apa yang di maksud."

    answered_chats = ChatHistory.query.filter(
        ChatHistory.client_id == client_id,
        ChatHistory.bot_response != default_unanswered_msg
    ).count()

    unanswered_chats = ChatHistory.query.filter(
        ChatHistory.client_id == client_id,
        ChatHistory.bot_response == default_unanswered_msg
    ).count()


    # Total FAQ count
    total_faqs = ChatbotResponse.query.filter_by(client_id=client_id).count()

    return render_template("analytics.html", 
         total_chats=total_chats,
         answered_chats=answered_chats,
         unanswered_chats=unanswered_chats,
         most_asked=most_asked,
         chat_activity=chat_activity,
         total_faqs=total_faqs)

@app.route("/healthz")
def healthz():
    return "OK"

# === Super Admin Approval ===
@app.route("/superadmin")
@admin_required
def superadmin():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    pending_clients = Client.query.filter_by(is_approved=False).all()
    return render_template("superadmin.html", pending_clients=pending_clients)

@app.route("/approve/<int:client_id>")
@admin_required
def approve_client(client_id):
    client = Client.query.get_or_404(client_id)
    client.is_approved = True
    db.session.commit()
    flash(f"Akun {client.business_name} telah disetujui!", "success")
    return redirect(url_for("superadmin"))


@app.route("/superadmin/clients")
def all_clients():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    clients = Client.query.all()
    return render_template("all_clients.html", clients=clients)

@app.route("/export-faq")
def export_faq():
    if "client_id" not in session:
        return redirect(url_for("login"))
    output = io.StringIO()
    writer = csv.writer(output)
    faqs = ChatbotResponse.query.filter_by(client_id=session["client_id"]).all()
    for f in faqs:
        writer.writerow([f.question, f.answer, f.category])
    output.seek(0)
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=faq.csv"}
    )

@app.before_request
def check_approval():
    allowed = ["login", "register", "static", "logout", "chat", "public_chat", "healthz"]
    if request.endpoint and any(request.endpoint.startswith(a) for a in allowed):
        return
    client_id = session.get("client_id")
    if client_id:
        client = Client.query.get(client_id)
        if client and not client.is_approved:
            flash("Akun Anda belum disetujui.", "warning")
            return redirect(url_for("logout"))

# === Setup DB dan admin ===
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not Client.query.filter_by(username="admin").first():
            admin = Client(
                username="admin",
                password=generate_password_hash("admin123"),
                role="admin",
                business_name="Admin",
                whatsapp="0000000000",
                is_approved=True  # penting agar tidak diblok saat login
            )
            db.session.add(admin)
            db.session.commit()
            print("Admin default dibuat.")
    app.run(debug=True)