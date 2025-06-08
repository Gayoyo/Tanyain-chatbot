from gabot import db, ChatbotResponse, app

with app.app_context():
    db.create_all()

    # Tambah data contoh
    faq1 = ChatbotResponse(question="Apa itu Chatbot?", answer="Chatbot adalah program yang dapat mensimulasikan percakapan dengan manusia.")
    faq2 = ChatbotResponse(question="Siapa yang membuat chatbot ini?", answer="Chatbot ini dibuat oleh tim developer.")
    faq3 = ChatbotResponse(question="Bagaimana cara kerja chatbot ini?", answer="Chatbot ini menggunakan TF-IDF dan cosine similarity untuk mencocokkan pertanyaan.")

    db.session.add_all([faq1, faq2, faq3])
    db.session.commit()
    print("Database berhasil diisi.")
 