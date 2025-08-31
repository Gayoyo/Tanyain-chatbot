# decorators.py
from flask import abort
from functools import wraps
from flask import session, redirect, url_for, flash

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            abort(404)
        return f(*args, **kwargs)
    return wrapper

def client_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("role") != "client":
            flash("Hanya klien yang bisa mengakses halaman ini.", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper
