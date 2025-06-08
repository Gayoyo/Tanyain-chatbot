import sqlite3

# Koneksi ke database
conn = sqlite3.connect("chatbot.db")
c = conn.cursor()

# Buat tabel clients (jika belum ada)
c.execute('''
CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    password TEXT
)
''')

# Buat tabel faq (dihubungkan ke client_id)
c.execute('''
CREATE TABLE IF NOT EXISTS faq (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category TEXT,
    FOREIGN KEY (client_id) REFERENCES clients(id)
)
''')

conn.commit()
conn.close()

print("Tabel clients dan faq berhasil dibuat (jika belum ada).")
