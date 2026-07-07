import json
import os
import pickle
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from datetime import datetime
from functools import lru_cache, wraps

import joblib
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_PROFILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PROFILE_IMAGE_BYTES = 3 * 1024 * 1024
PROFILE_IMAGE_PACKET_MARGIN_BYTES = 64 * 1024
REMOTE_IMAGE_FETCH_TIMEOUT_SECONDS = 8
PROFILE_THEME_OPTIONS = {
    "neon-purple": "Neon Purple",
    "ocean-blue": "Ocean Blue",
    "sunset-orange": "Sunset Orange",
    "emerald-green": "Emerald Green",
}
MARITAL_STATUS_OPTIONS = {
    "single": "Single",
    "married": "Married",
}
PASSWORD_RESET_STATUSES = ("Pending", "Approved", "Rejected", "Completed")

MODEL_NUMERIC_COLUMNS = [
    "Built_Up_SF",
    "Bathroom",
    "Furnishing",
    "Bedroom",
    "Tenure",
    "Car_Park",
    "Property_Type",
    "Land_Size",
    "Unit_Type",
]

# Fallback if model_columns.pkl is unavailable.
DEFAULT_FEATURE_COLUMNS = MODEL_NUMERIC_COLUMNS + [
    "negeri_Kedah",
    "negeri_Kelantan",
    "negeri_Malacca",
    "negeri_Negeri Sembilan",
    "negeri_Pahang",
    "negeri_Penang",
    "negeri_Perak",
    "negeri_Perlis",
    "negeri_Putrajaya",
    "negeri_Sabah",
    "negeri_Sarawak",
    "negeri_Selangor",
    "negeri_Terengganu",
]

FURNISHING_OPTIONS = {
    0: "Unknown",
    1: "Partly Furnished",
    2: "Unfurnished",
    3: "Fully Furnished",
}
TENURE_OPTIONS = {0: "Unknown", 1: "Leasehold", 2: "Freehold"}
PROPERTY_TYPE_OPTIONS = {
    0: "Unknown",
    1: "Terrace House",
    2: "Link Bungalow / Semi-Detached House",
    3: "Condominium / Apartment / Serviced Residence",
    4: "Flat",
    5: "Bungalow / Detached House / Villa",
    6: "Townhouse",
    7: "Cluster",
    8: "Low-Cost House",
    9: "Superlink",
    10: "Penthouse",
    11: "Commercial / Non-Residential",
    12: "Land",
}
UNIT_TYPE_OPTIONS = {0: "Unknown", 1: "Intermediate Lot", 2: "Corner Lot", 3: "End Lot"}
XGBOOST_AFTER_TUNED_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

PROPERTY_DATASET_PATH = os.getenv(
    "PROPERTY_DATASET_PATH",
    r"c:\Users\user\Downloads\dataset_with_negeri_filled_with_src.csv",
)
PROPERTY_SRC_DATASET_PATH = os.getenv(
    "PROPERTY_SRC_DATASET_PATH",
    r"c:\Users\user\Downloads\dataset_with_negeri_filled_with_src.csv",
)
HOSPITAL_DATASET_PATH = os.getenv(
    "HOSPITAL_DATASET_PATH",
    r"c:\Users\user\Desktop\FYP\hospital_final_geo.csv",
)
SECONDARY_SCHOOL_DATASET_PATH = os.getenv(
    "SECONDARY_SCHOOL_DATASET_PATH",
    r"c:\Users\user\Desktop\FYP\Senarai Sekolah Menengah Kementerian Pendidikan Malaysia\2022 senarai sekolah menengah    .csv",
)
PRIMARY_SCHOOL_DATASET_PATH = os.getenv(
    "PRIMARY_SCHOOL_DATASET_PATH",
    r"c:\Users\user\Desktop\FYP\Senarai Sekolah Rendah Kementerian Pendidikan Malaysia\2022 senarai sekolah rendah    .csv",
)
PROPERTY_IMAGE_DIR = os.getenv(
    "PROPERTY_IMAGE_DIR",
    r"c:\Users\user\Desktop\FYP\durianprop_images",
)
ADMIN_DEFAULT_EMAIL = os.getenv("ADMIN_EMAIL", "admin@spv.local")
ADMIN_DEFAULT_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_DEFAULT_NAME = os.getenv("ADMIN_NAME", "System Administrator")

SEED_PROPERTY_LISTINGS = [
    {
        "title": "Modern Terrace Home, Shah Alam",
        "area": "Shah Alam",
        "negeri": "Selangor",
        "property_type": 1,
        "built_up_sf": 1650,
        "land_size": 1400,
        "bedroom": 4,
        "bathroom": 3,
        "car_park": 2,
        "furnishing": 1,
        "tenure": 2,
        "unit_type": 1,
        "listing_price": 620000,
        "latitude": 3.0738,
        "longitude": 101.5183,
        "description": "Near schools and access to major highways.",
    },
    {
        "title": "Family Condo, Petaling Jaya",
        "area": "Petaling Jaya",
        "negeri": "Selangor",
        "property_type": 3,
        "built_up_sf": 1100,
        "land_size": 1100,
        "bedroom": 3,
        "bathroom": 2,
        "car_park": 2,
        "furnishing": 3,
        "tenure": 1,
        "unit_type": 0,
        "listing_price": 540000,
        "latitude": 3.1073,
        "longitude": 101.6067,
        "description": "Serviced residence with nearby LRT and mall.",
    },
    {
        "title": "Corner Lot Link House, Seremban",
        "area": "Seremban",
        "negeri": "Negeri Sembilan",
        "property_type": 2,
        "built_up_sf": 1900,
        "land_size": 2100,
        "bedroom": 4,
        "bathroom": 3,
        "car_park": 3,
        "furnishing": 2,
        "tenure": 2,
        "unit_type": 2,
        "listing_price": 590000,
        "latitude": 2.7297,
        "longitude": 101.9381,
        "description": "Quiet neighborhood with larger land area.",
    },
    {
        "title": "City Apartment, Penang",
        "area": "George Town",
        "negeri": "Penang",
        "property_type": 3,
        "built_up_sf": 900,
        "land_size": 900,
        "bedroom": 3,
        "bathroom": 2,
        "car_park": 1,
        "furnishing": 1,
        "tenure": 1,
        "unit_type": 0,
        "listing_price": 480000,
        "latitude": 5.4141,
        "longitude": 100.3288,
        "description": "Strategic location near city center and amenities.",
    },
    {
        "title": "Freehold Terrace, Johor Bahru",
        "area": "Johor Bahru",
        "negeri": "Johor",
        "property_type": 1,
        "built_up_sf": 1750,
        "land_size": 1500,
        "bedroom": 4,
        "bathroom": 3,
        "car_park": 2,
        "furnishing": 1,
        "tenure": 2,
        "unit_type": 1,
        "listing_price": 520000,
        "latitude": 1.4927,
        "longitude": 103.7414,
        "description": "Suitable for first-time buyers and families.",
    },
    {
        "title": "High-Rise Unit, Kuala Terengganu",
        "area": "Kuala Terengganu",
        "negeri": "Terengganu",
        "property_type": 3,
        "built_up_sf": 980,
        "land_size": 980,
        "bedroom": 3,
        "bathroom": 2,
        "car_park": 1,
        "furnishing": 2,
        "tenure": 1,
        "unit_type": 0,
        "listing_price": 310000,
        "latitude": 5.3296,
        "longitude": 103.1370,
        "description": "Affordable option near public transport.",
    },
]


def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "a202336"),
        port=int(os.getenv("DB_PORT", "3306")),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def _safe_next_url(next_url):
    if not next_url or not next_url.startswith("/") or next_url.startswith("//"):
        return url_for("home")
    return next_url


def _redirect_admin_to_dashboard():
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    return None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, email, full_name, contact_number, address, postcode, state,
                       marital_status, family_count, profile_image_url,
                       (profile_image_blob IS NOT NULL) AS has_profile_image_blob,
                       profile_theme, avatar_size, user_role, created_at
                FROM users
                WHERE id = %s
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            has_blob = bool(_safe_int(row.get("has_profile_image_blob"), default=0))
            if has_blob:
                row["profile_image_src"] = url_for("profile_image_by_user", user_id=row["id"])
            else:
                row["profile_image_src"] = None
            return row
    finally:
        conn.close()


def get_user_by_email(email):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            return cursor.fetchone()
    finally:
        conn.close()


def create_user(email, password, full_name="", user_role="user"):
    password_hash = generate_password_hash(password)
    normalized_role = _safe_str(user_role, "user").strip().lower()
    if normalized_role not in {"user", "admin"}:
        normalized_role = "user"
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (email, password_hash, full_name, user_role)
                VALUES (%s, %s, %s, %s)
                """,
                (email, password_hash, full_name or None, normalized_role),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def _normalize_password_reset_status(raw_status):
    status = _safe_str(raw_status, "Pending").strip().title()
    if status not in PASSWORD_RESET_STATUSES:
        return "Pending"
    return status


def create_password_reset_request(email, reason=""):
    user = get_user_by_email(email)
    if not user:
        return None
    user_id = _safe_int(user.get("id"), default=0)
    customer_name = _safe_str(user.get("full_name"), "").strip() or _safe_str(user.get("email"), "Customer")
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO password_reset_requests (user_id, customer_name, email, reason, status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    user_id if user_id > 0 else None,
                    customer_name,
                    email,
                    _safe_str(reason, "").strip() or None,
                    "Pending",
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_latest_password_reset_request(email):
    clean_email = _safe_str(email, "").strip().lower()
    if not clean_email:
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id, user_id, customer_name, email, reason, status,
                    request_date, admin_action_date, admin_action_by, completed_at
                FROM password_reset_requests
                WHERE email = %s
                ORDER BY request_date DESC, id DESC
                LIMIT 1
                """,
                (clean_email,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            row["status"] = _normalize_password_reset_status(row.get("status"))
            return row
    finally:
        conn.close()


def fetch_password_reset_requests(search_text="", status_filter="all"):
    search = _safe_str(search_text, "").strip().lower()
    status = _safe_str(status_filter, "all").strip().title()
    conn = get_db_connection()
    rows = []
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id, user_id, customer_name, email, reason, status,
                    request_date, admin_action_date, admin_action_by, completed_at
                FROM password_reset_requests
                ORDER BY request_date DESC, id DESC
                LIMIT 4000
                """
            )
            db_rows = cursor.fetchall()
        for row in db_rows:
            row_status = _normalize_password_reset_status(row.get("status"))
            if status != "All" and row_status != status:
                continue
            if search:
                haystack = " ".join(
                    [
                        str(row.get("id", "")),
                        _safe_str(row.get("customer_name"), ""),
                        _safe_str(row.get("email"), ""),
                        _safe_str(row.get("reason"), ""),
                    ]
                ).lower()
                if search not in haystack:
                    continue
            rows.append(
                {
                    "id": _safe_int(row.get("id"), default=0),
                    "user_id": _safe_int(row.get("user_id"), default=0),
                    "customer_name": _safe_str(row.get("customer_name"), "").strip() or "Customer",
                    "email": _safe_str(row.get("email"), "-"),
                    "reason": _safe_str(row.get("reason"), "-") or "-",
                    "status": row_status,
                    "request_date": row.get("request_date"),
                    "request_date_label": _admin_datetime_label(row.get("request_date"), include_time=True),
                    "admin_action_date": row.get("admin_action_date"),
                    "admin_action_date_label": _admin_datetime_label(
                        row.get("admin_action_date"), include_time=True
                    ),
                    "admin_action_by": _safe_str(row.get("admin_action_by"), "-") or "-",
                }
            )
    finally:
        conn.close()
    return rows


def count_pending_password_reset_requests():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) AS total
                FROM password_reset_requests
                WHERE status = 'Pending'
                """
            )
            return _safe_int(cursor.fetchone().get("total"), default=0)
    finally:
        conn.close()


def get_password_reset_request_by_id(request_id):
    rid = _safe_int(request_id, default=0)
    if rid <= 0:
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id, user_id, customer_name, email, reason, status,
                    request_date, admin_action_date, admin_action_by, completed_at
                FROM password_reset_requests
                WHERE id = %s
                LIMIT 1
                """,
                (rid,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": _safe_int(row.get("id"), default=0),
                "user_id": _safe_int(row.get("user_id"), default=0),
                "customer_name": _safe_str(row.get("customer_name"), "").strip() or "Customer",
                "email": _safe_str(row.get("email"), "-"),
                "reason": _safe_str(row.get("reason"), "-") or "-",
                "status": _normalize_password_reset_status(row.get("status")),
                "request_date_label": _admin_datetime_label(row.get("request_date"), include_time=True),
                "admin_action_date_label": _admin_datetime_label(
                    row.get("admin_action_date"), include_time=True
                ),
                "admin_action_by": _safe_str(row.get("admin_action_by"), "-") or "-",
            }
    finally:
        conn.close()


def update_password_reset_request_status(request_id, status, action_by=""):
    rid = _safe_int(request_id, default=0)
    target_status = _normalize_password_reset_status(status)
    if rid <= 0:
        return False
    if target_status not in {"Approved", "Rejected"}:
        return False
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT status
                FROM password_reset_requests
                WHERE id = %s
                LIMIT 1
                """,
                (rid,),
            )
            current = cursor.fetchone()
            if not current:
                return False
            current_status = _normalize_password_reset_status(current.get("status"))
            if current_status != "Pending":
                return False
            cursor.execute(
                """
                UPDATE password_reset_requests
                SET status = %s,
                    admin_action_date = CURRENT_TIMESTAMP,
                    admin_action_by = %s
                WHERE id = %s
                """,
                (target_status, _safe_str(action_by, "").strip() or None, rid),
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def _ensure_column_exists(cursor, table_name, column_name, column_definition):
    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (table_name, column_name),
    )
    if int(cursor.fetchone()["total"]) == 0:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")


def _save_profile_image(file_storage):
    if not file_storage or not file_storage.filename:
        return None

    original_name = secure_filename(file_storage.filename)
    extension = os.path.splitext(original_name)[1].lower()
    if extension not in ALLOWED_PROFILE_EXTENSIONS:
        raise ValueError("Profile image must be JPG, PNG, or WEBP.")
    image_bytes = file_storage.read()
    if not image_bytes:
        raise ValueError("Uploaded profile image is empty.")
    if len(image_bytes) > MAX_PROFILE_IMAGE_BYTES:
        raise ValueError("Profile image must be smaller than 3 MB.")
    mime_type = _safe_str(file_storage.mimetype, "").strip().lower()
    allowed_mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    expected_mime = allowed_mime_map.get(extension, "")
    if not mime_type:
        mime_type = expected_mime
    if expected_mime and mime_type != expected_mime:
        raise ValueError("Profile image type does not match file extension.")
    return {
        "blob": image_bytes,
        "mime": mime_type or "application/octet-stream",
        "name": original_name[:255] or f"profile{extension}",
    }


def _fetch_db_max_allowed_packet_bytes(cursor):
    try:
        cursor.execute("SHOW VARIABLES LIKE 'max_allowed_packet'")
        row = cursor.fetchone() or {}
        return _safe_int(row.get("Value"), default=0)
    except Exception:
        return 0


def _resolve_default_profile_image_path():
    asset_dirs = [os.path.join(BASE_DIR, "assets")]
    base_slug = re.sub(r"[^A-Za-z0-9]+", "-", os.path.abspath(BASE_DIR)).strip("-")
    user_home = os.path.expanduser("~")
    cursor_project_dir = os.path.join(user_home, ".cursor", "projects")
    if os.path.isdir(cursor_project_dir):
        asset_dirs.append(os.path.join(cursor_project_dir, base_slug, "assets"))
        asset_dirs.append(os.path.join(cursor_project_dir, base_slug.lower(), "assets"))

    candidates = []
    for assets_dir in asset_dirs:
        if not os.path.isdir(assets_dir):
            continue
        for file_name in os.listdir(assets_dir):
            lower_name = file_name.lower()
            if not lower_name.endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            if "no_profile" in lower_name or "no-image" in lower_name or "no_image" in lower_name:
                file_path = os.path.join(assets_dir, file_name)
                if os.path.isfile(file_path):
                    candidates.append(file_path)

    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


@app.context_processor
def inject_auth_context():
    user = get_current_user() if session.get("user_id") else None
    active_site_theme = user.get("profile_theme") if user else "neon-purple"
    if active_site_theme not in PROFILE_THEME_OPTIONS:
        active_site_theme = "neon-purple"
    return {
        "current_user": user,
        "active_site_theme": active_site_theme,
        "profile_theme_options": PROFILE_THEME_OPTIONS,
        "marital_status_options": MARITAL_STATUS_OPTIONS,
        "state_choices": get_state_choices(),
    }


def load_serialized_object(file_path):
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as file:
            return pickle.load(file)


def _resolve_project_path(path_value):
    clean = _safe_str(path_value, "")
    if not clean:
        return clean
    if os.path.isabs(clean):
        return clean
    return os.path.join(BASE_DIR, clean)


def _is_direct_image_url(url_value):
    url_text = _safe_str(url_value, "").strip().lower()
    if not url_text.startswith(("http://", "https://")):
        return False
    image_suffixes = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif")
    if any(url_text.endswith(suffix) for suffix in image_suffixes):
        return True
    if "?" in url_text:
        path_only = url_text.split("?", 1)[0]
        return any(path_only.endswith(suffix) for suffix in image_suffixes)
    return False


def _normalize_listing_url(url_value):
    clean_url = _safe_str(url_value, "").strip()
    if not clean_url.startswith(("http://", "https://")):
        return clean_url
    try:
        parsed = urllib.parse.urlsplit(clean_url)
    except Exception:
        return clean_url

    host = _safe_str(parsed.netloc, "").lower()
    if host == "www.durianproperty.com.my":
        return urllib.parse.urlunsplit(
            (parsed.scheme or "https", "durianproperty.com.my", parsed.path, parsed.query, parsed.fragment)
        )
    return clean_url


@lru_cache(maxsize=2000)
def _extract_image_url_from_listing_page(page_url):
    clean_url = _normalize_listing_url(page_url)
    if not clean_url.startswith(("http://", "https://")):
        return None
    request_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    try:
        request_obj = urllib.request.Request(clean_url, headers=request_headers)
        with urllib.request.urlopen(request_obj, timeout=REMOTE_IMAGE_FETCH_TIMEOUT_SECONDS) as response:
            content_bytes = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None

    try:
        html_text = content_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return None

    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = unescape(_safe_str(match.group(1), "").strip())
        if candidate.startswith("//"):
            candidate = "https:" + candidate
        if _is_direct_image_url(candidate):
            return candidate
    return None


def _resolve_property_image_url_for_display(property_id, raw_image_url):
    clean_url = _safe_str(raw_image_url, "").strip()
    if not clean_url:
        return None
    if _is_direct_image_url(clean_url):
        return clean_url

    resolved = _extract_image_url_from_listing_page(clean_url)
    if not resolved:
        return None

    pid = _safe_int(property_id, default=0)
    if pid > 0 and resolved != clean_url:
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE property_listings
                    SET image_url = %s
                    WHERE id = %s
                    """,
                    (resolved, pid),
                )
        except Exception:
            pass
        finally:
            if conn:
                conn.close()
    return resolved


@lru_cache(maxsize=1)
def load_model():
    raw_model_path = os.getenv("MODEL_PATH", "models/best_model.pkl")
    model_path = _resolve_project_path(raw_model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please copy your trained model there or update MODEL_PATH in .env."
        )
    return load_serialized_object(model_path)


@lru_cache(maxsize=1)
def load_feature_columns():
    raw_columns_path = os.getenv("FEATURE_COLUMNS_PATH", "models/model_columns.pkl")
    columns_path = _resolve_project_path(raw_columns_path)
    if not os.path.exists(columns_path):
        return DEFAULT_FEATURE_COLUMNS

    columns = load_serialized_object(columns_path)
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError("FEATURE_COLUMNS_PATH must contain a non-empty list of columns.")
    return [str(col) for col in columns]


@lru_cache(maxsize=1)
def get_state_choices():
    columns = load_feature_columns()
    encoded_states = [
        col.replace("negeri_", "", 1) for col in columns if col.startswith("negeri_")
    ]
    base_state = os.getenv("BASE_STATE", "Johor")
    return sorted(set([base_state] + encoded_states))


def _safe_str(value, default=""):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value).strip()


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_listing_price(value):
    """Keep listing_price aligned with MySQL DECIMAL(15,2) values."""
    return round(_safe_float(value, 0.0), 2)


def normalize_property_row(row):
    item = dict(row)
    if "listing_price" in item:
        item["listing_price"] = normalize_listing_price(item.get("listing_price"))
    return item


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _get_llm_runtime_config():
    deepseek_key = _safe_str(os.getenv("DEEPSEEK_API_KEY", ""), "")
    deepseek_url = _safe_str(
        os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"),
        "",
    )
    deepseek_model = _safe_str(os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), "")
    if deepseek_key:
        return {
            "api_key": deepseek_key,
            "api_url": deepseek_url,
            "model_name": deepseek_model or "deepseek-chat",
        }

    return {
        "api_key": _safe_str(os.getenv("LLM_API_KEY", ""), ""),
        "api_url": _safe_str(os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions"), ""),
        "model_name": _safe_str(os.getenv("LLM_MODEL", "gpt-4o-mini"), ""),
    }


def _llm_is_configured():
    llm_cfg = _get_llm_runtime_config()
    api_key = llm_cfg["api_key"]
    api_url = llm_cfg["api_url"]
    model_name = llm_cfg["model_name"]
    return bool(api_key and api_url and model_name)


def _extract_first_json_object(text):
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM response does not contain a valid JSON object.")


def _ai_match_text_signature(ai_match):
    if not isinstance(ai_match, dict):
        return ""

    summary = " ".join(_safe_str(ai_match.get("summary"), "").lower().split())
    tips = ai_match.get("suggestions")
    if not isinstance(tips, list):
        tips = []

    parts = [summary]
    for tip in tips[:3]:
        clean_tip = " ".join(_safe_str(tip, "").lower().split())
        if clean_tip:
            parts.append(clean_tip)
    return "|".join([part for part in parts if part])


def _call_llm_chat_json(system_prompt, user_prompt, temperature=0.9, max_tokens=320):
    llm_cfg = _get_llm_runtime_config()
    api_key = llm_cfg["api_key"]
    api_url = llm_cfg["api_url"]
    model_name = llm_cfg["model_name"]
    timeout_seconds = max(3.0, _safe_float(os.getenv("LLM_TIMEOUT_SECONDS", "15"), 15.0))

    if not api_key or not api_url or not model_name:
        raise RuntimeError("LLM API is not configured.")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": max(0.2, min(1.3, float(temperature))),
        "max_tokens": max(120, _safe_int(max_tokens, 320)),
    }
    request_obj = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API request failed (HTTP {exc.code}): {error_body[:240]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc

    data = json.loads(response_text)
    choices = data.get("choices") or []
    content = ""
    if choices:
        content = _safe_str((choices[0].get("message") or {}).get("content"), "")
    if not content:
        raise ValueError("LLM API response did not return message content.")
    return _extract_first_json_object(content)


def _normalize_text_state(value):
    raw = _safe_str(value, default="")
    return " ".join(raw.split()).title() if raw else "Unknown"


def _haversine_km(lat1, lon1, lat2_series, lon2_series):
    # Calculate geodesic distances using vectorized Haversine formula.
    earth_radius_km = 6371.0
    lat1_rad = np.radians(float(lat1))
    lon1_rad = np.radians(float(lon1))
    lat2_rad = np.radians(lat2_series.astype(float).to_numpy())
    lon2_rad = np.radians(lon2_series.astype(float).to_numpy())
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius_km * c


def _read_csv_safely(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1")


def _build_google_maps_link(latitude, longitude):
    return f"https://www.google.com/maps/search/?api=1&query={float(latitude)},{float(longitude)}"


@lru_cache(maxsize=1)
def load_hospital_dataset():
    df = _read_csv_safely(HOSPITAL_DATASET_PATH)
    if df.empty:
        return df
    expected_cols = {"hospital_name", "address", "negeri", "latitude", "longitude"}
    if not expected_cols.issubset(set(df.columns)):
        return pd.DataFrame()

    clean = df.copy()
    clean["latitude"] = pd.to_numeric(clean["latitude"], errors="coerce")
    clean["longitude"] = pd.to_numeric(clean["longitude"], errors="coerce")
    clean = clean.dropna(subset=["latitude", "longitude"])
    clean["name"] = clean["hospital_name"].fillna("Unknown Hospital")
    clean["address_text"] = clean["address"].fillna("")
    clean["state_text"] = clean["negeri"].fillna("")
    return clean[["name", "address_text", "state_text", "latitude", "longitude"]]


def _load_school_dataset(file_path):
    df = _read_csv_safely(file_path)
    if df.empty:
        return df
    expected_cols = {"NAMASEKOLAH", "ALAMATSURAT", "BANDARSURAT", "NEGERI", "KOORDINATXX", "KOORDINATYY"}
    if not expected_cols.issubset(set(df.columns)):
        return pd.DataFrame()

    clean = df.copy()
    clean["longitude"] = pd.to_numeric(clean["KOORDINATXX"], errors="coerce")
    clean["latitude"] = pd.to_numeric(clean["KOORDINATYY"], errors="coerce")
    clean = clean.dropna(subset=["latitude", "longitude"])
    clean["name"] = clean["NAMASEKOLAH"].fillna("Unknown School")
    clean["address_text"] = (
        clean["ALAMATSURAT"].fillna("").astype(str).str.strip()
        + ", "
        + clean["BANDARSURAT"].fillna("").astype(str).str.strip()
    ).str.strip(", ")
    clean["state_text"] = clean["NEGERI"].fillna("").astype(str).str.title()
    return clean[["name", "address_text", "state_text", "latitude", "longitude"]]


@lru_cache(maxsize=1)
def load_secondary_school_dataset():
    return _load_school_dataset(SECONDARY_SCHOOL_DATASET_PATH)


@lru_cache(maxsize=1)
def load_primary_school_dataset():
    return _load_school_dataset(PRIMARY_SCHOOL_DATASET_PATH)


def find_nearest_place(property_lat, property_lon, places_df):
    if places_df.empty:
        return None
    distances = _haversine_km(property_lat, property_lon, places_df["latitude"], places_df["longitude"])
    min_idx = int(np.argmin(distances))
    row = places_df.iloc[min_idx]
    return {
        "name": _safe_str(row["name"], "Unknown"),
        "address": _safe_str(row["address_text"], "-"),
        "negeri": _safe_str(row["state_text"], "-"),
        "distance_km": round(float(distances[min_idx]), 3),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "google_maps_url": _build_google_maps_link(row["latitude"], row["longitude"]),
    }


def load_property_rows_from_csv(limit_rows=20000):
    df = _read_csv_safely(PROPERTY_DATASET_PATH)
    if df.empty:
        return []

    required = {
        "Price",
        "Built_Up_SF",
        "Bathroom",
        "Furnishing",
        "Bedroom",
        "Tenure",
        "Car_Park",
        "Area",
        "Property_Type",
        "Unit_Type",
        "Land_Size",
        "Latitude",
        "Longitude",
        "negeri",
    }
    if not required.issubset(set(df.columns)):
        return []

    rows = []
    for _, item in df.head(limit_rows).iterrows():
        title = _safe_str(item.get("Property_Address")) or _safe_str(item.get("Property_ID")) or "Property"
        area = _safe_str(item.get("Area"), default="Unknown Area")
        negeri = _normalize_text_state(item.get("negeri"))
        listing_price = max(0.0, _safe_float(item.get("Price"), default=0.0))
        built_up = max(0.0, _safe_float(item.get("Built_Up_SF"), default=0.0))
        land_size = max(0.0, _safe_float(item.get("Land_Size"), default=built_up))
        description = _safe_str(item.get("Desc"), default="")[:6000]
        image_url = _safe_str(item.get("src"), default="").strip()
        if image_url and not image_url.lower().startswith(("http://", "https://")):
            image_url = ""
        lat_value = _safe_float(item.get("Latitude"), default=np.nan)
        lon_value = _safe_float(item.get("Longitude"), default=np.nan)
        latitude = None if pd.isna(lat_value) else lat_value
        longitude = None if pd.isna(lon_value) else lon_value

        rows.append(
            {
                "title": title[:200],
                "area": area[:100],
                "negeri": negeri[:100],
                "property_type": _safe_int(item.get("Property_Type"), default=0),
                "built_up_sf": built_up,
                "land_size": land_size,
                "bedroom": max(0, _safe_int(item.get("Bedroom"), default=0)),
                "bathroom": max(0, _safe_int(item.get("Bathroom"), default=0)),
                "car_park": max(0, _safe_int(item.get("Car_Park"), default=0)),
                "furnishing": _safe_int(item.get("Furnishing"), default=0),
                "tenure": _safe_int(item.get("Tenure"), default=0),
                "unit_type": _safe_int(item.get("Unit_Type"), default=0),
                "listing_price": listing_price,
                "latitude": latitude,
                "longitude": longitude,
                "description": description,
                "image_url": image_url or None,
            }
        )
    return rows


def import_property_dataset_into_db(replace_existing=False):
    rows = load_property_rows_from_csv()
    if not rows:
        raise ValueError("Property dataset is missing/invalid. Check PROPERTY_DATASET_PATH.")

    conn = get_db_connection()
    inserted = 0
    try:
        with conn.cursor() as cursor:
            if replace_existing:
                cursor.execute("TRUNCATE TABLE property_listings")
            cursor.executemany(
                """
                INSERT INTO property_listings (
                    title, area, negeri, property_type, built_up_sf, land_size,
                    bedroom, bathroom, car_park, furnishing, tenure, unit_type,
                    listing_price, latitude, longitude, description, image_url
                )
                VALUES (
                    %(title)s, %(area)s, %(negeri)s, %(property_type)s, %(built_up_sf)s,
                    %(land_size)s, %(bedroom)s, %(bathroom)s, %(car_park)s,
                    %(furnishing)s, %(tenure)s, %(unit_type)s, %(listing_price)s,
                    %(latitude)s, %(longitude)s, %(description)s, %(image_url)s
                )
                """,
                rows,
            )
            inserted = len(rows)
    finally:
        conn.close()
    return inserted


def ensure_database_tables():
    create_predictions_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        input_json JSON NOT NULL,
        predicted_price DECIMAL(15,2) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    create_property_sql = """
    CREATE TABLE IF NOT EXISTS property_listings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(200) NOT NULL,
        area VARCHAR(100) NOT NULL,
        negeri VARCHAR(100) NOT NULL,
        property_type INT NOT NULL,
        built_up_sf DECIMAL(10,2) NOT NULL,
        land_size DECIMAL(10,2) NOT NULL,
        bedroom INT NOT NULL,
        bathroom INT NOT NULL,
        car_park INT NOT NULL,
        furnishing INT NOT NULL,
        tenure INT NOT NULL,
        unit_type INT NOT NULL,
        listing_price DECIMAL(15,2) NOT NULL,
        latitude DECIMAL(10,6) NULL,
        longitude DECIMAL(10,6) NULL,
        description TEXT NULL,
        image_url VARCHAR(1200) NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    create_users_sql = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        full_name VARCHAR(120) NULL,
        contact_number VARCHAR(30) NULL,
        address TEXT NULL,
        postcode VARCHAR(10) NULL,
        state VARCHAR(100) NULL,
        marital_status VARCHAR(20) NULL,
        family_count INT NULL,
        profile_image_url VARCHAR(1200) NULL,
        profile_image_blob LONGBLOB NULL,
        profile_image_mime VARCHAR(80) NULL,
        profile_image_name VARCHAR(255) NULL,
        user_role VARCHAR(20) NOT NULL DEFAULT 'user',
        profile_theme VARCHAR(40) NOT NULL DEFAULT 'neon-purple',
        avatar_size INT NOT NULL DEFAULT 120,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    );
    """
    create_password_reset_requests_sql = """
    CREATE TABLE IF NOT EXISTS password_reset_requests (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NULL,
        customer_name VARCHAR(120) NULL,
        email VARCHAR(255) NOT NULL,
        reason TEXT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'Pending',
        request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        admin_action_date TIMESTAMP NULL,
        admin_action_by VARCHAR(255) NULL,
        completed_at TIMESTAMP NULL,
        INDEX idx_password_reset_email (email),
        INDEX idx_password_reset_status (status),
        INDEX idx_password_reset_date (request_date)
    );
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(create_predictions_sql)
            cursor.execute(create_property_sql)
            cursor.execute(create_users_sql)
            cursor.execute(create_password_reset_requests_sql)
            _ensure_column_exists(cursor, "predictions", "user_id", "user_id INT NULL")
            _ensure_column_exists(cursor, "users", "postcode", "postcode VARCHAR(10) NULL")
            _ensure_column_exists(cursor, "users", "state", "state VARCHAR(100) NULL")
            _ensure_column_exists(
                cursor,
                "users",
                "profile_image_url",
                "profile_image_url VARCHAR(1200) NULL",
            )
            _ensure_column_exists(
                cursor,
                "users",
                "profile_image_blob",
                "profile_image_blob LONGBLOB NULL",
            )
            _ensure_column_exists(
                cursor,
                "users",
                "profile_image_mime",
                "profile_image_mime VARCHAR(80) NULL",
            )
            _ensure_column_exists(
                cursor,
                "users",
                "profile_image_name",
                "profile_image_name VARCHAR(255) NULL",
            )
            _ensure_column_exists(
                cursor,
                "users",
                "user_role",
                "user_role VARCHAR(20) NOT NULL DEFAULT 'user'",
            )
            _ensure_column_exists(
                cursor,
                "password_reset_requests",
                "customer_name",
                "customer_name VARCHAR(120) NULL",
            )
            _ensure_column_exists(
                cursor,
                "password_reset_requests",
                "admin_action_date",
                "admin_action_date TIMESTAMP NULL",
            )
            _ensure_column_exists(
                cursor,
                "password_reset_requests",
                "admin_action_by",
                "admin_action_by VARCHAR(255) NULL",
            )
            _ensure_column_exists(
                cursor,
                "password_reset_requests",
                "completed_at",
                "completed_at TIMESTAMP NULL",
            )
            cursor.execute(
                """
                UPDATE users
                SET user_role = 'user'
                WHERE user_role IS NULL OR user_role = ''
                """
            )
            _ensure_column_exists(
                cursor,
                "property_listings",
                "image_url",
                "image_url VARCHAR(1200) NULL",
            )
            cursor.execute("SELECT COUNT(*) AS total FROM property_listings")
            if int(cursor.fetchone()["total"]) == 0:
                csv_rows = load_property_rows_from_csv()
                if csv_rows:
                    cursor.executemany(
                        """
                        INSERT INTO property_listings (
                            title, area, negeri, property_type, built_up_sf, land_size,
                            bedroom, bathroom, car_park, furnishing, tenure, unit_type,
                            listing_price, latitude, longitude, description, image_url
                        )
                        VALUES (
                            %(title)s, %(area)s, %(negeri)s, %(property_type)s, %(built_up_sf)s,
                            %(land_size)s, %(bedroom)s, %(bathroom)s, %(car_park)s,
                            %(furnishing)s, %(tenure)s, %(unit_type)s, %(listing_price)s,
                            %(latitude)s, %(longitude)s, %(description)s, %(image_url)s
                        )
                        """,
                        csv_rows,
                    )
                else:
                    seed_rows = [dict(item, image_url=None) for item in SEED_PROPERTY_LISTINGS]
                    cursor.executemany(
                        """
                        INSERT INTO property_listings (
                            title, area, negeri, property_type, built_up_sf, land_size,
                            bedroom, bathroom, car_park, furnishing, tenure, unit_type,
                            listing_price, latitude, longitude, description, image_url
                        )
                        VALUES (
                            %(title)s, %(area)s, %(negeri)s, %(property_type)s, %(built_up_sf)s,
                            %(land_size)s, %(bedroom)s, %(bathroom)s, %(car_park)s,
                            %(furnishing)s, %(tenure)s, %(unit_type)s, %(listing_price)s,
                            %(latitude)s, %(longitude)s, %(description)s, %(image_url)s
                        )
                        """,
                        seed_rows,
                    )
    finally:
        conn.close()


def _input_value_as_str(value, default=""):
    if value is None:
        return default
    return str(value).strip()


def parse_float(form_data, key, min_value=0.0, max_value=1_000_000.0):
    raw = _input_value_as_str(form_data.get(key, ""))
    if raw == "":
        raise ValueError(f"{key} is required.")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be numeric.") from exc
    if value < min_value or value > max_value:
        raise ValueError(f"{key} must be between {min_value} and {max_value}.")
    return value


def parse_int_choice(form_data, key, options):
    raw = _input_value_as_str(form_data.get(key, ""))
    if raw == "":
        raise ValueError(f"{key} is required.")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer option.") from exc
    if value not in options:
        raise ValueError(f"Invalid value for {key}.")
    return value


def payload_to_model_row(input_payload, feature_columns):
    row = {col: 0.0 for col in feature_columns}
    for key in MODEL_NUMERIC_COLUMNS:
        if key in row:
            row[key] = float(input_payload[key])

    negeri_value = input_payload.get("state") or input_payload.get("negeri")
    if not negeri_value:
        raise ValueError("State is required.")
    negeri_column = f"negeri_{negeri_value}"
    if negeri_column in row:
        row[negeri_column] = 1.0
    return row


def parse_prediction_form(form_data):
    feature_columns = load_feature_columns()
    payload = {
        "Built_Up_SF": parse_float(form_data, "Built_Up_SF", min_value=150.0, max_value=10000.0),
        "Bathroom": parse_float(form_data, "Bathroom", min_value=1.0, max_value=20.0),
        "Furnishing": parse_int_choice(form_data, "Furnishing", FURNISHING_OPTIONS),
        "Bedroom": parse_float(form_data, "Bedroom", min_value=1.0, max_value=20.0),
        "Tenure": parse_int_choice(form_data, "Tenure", TENURE_OPTIONS),
        "Car_Park": parse_float(form_data, "Car_Park", min_value=0.0, max_value=10.0),
        "Property_Type": parse_int_choice(form_data, "Property_Type", PROPERTY_TYPE_OPTIONS),
        "Land_Size": parse_float(form_data, "Land_Size", min_value=150.0, max_value=20000.0),
        "Unit_Type": parse_int_choice(form_data, "Unit_Type", UNIT_TYPE_OPTIONS),
        "area_text": _input_value_as_str(form_data.get("area_text", "")),
        "state": _input_value_as_str(form_data.get("state", "")),
    }
    if not payload["state"]:
        raise ValueError("State is required.")
    model_row = payload_to_model_row(payload, feature_columns)
    feature_df = pd.DataFrame([model_row], columns=feature_columns, dtype=float)
    return payload, feature_df


def save_prediction(input_payload, predicted_price, user_id=None):
    if not user_id:
        return
    db_payload = dict(input_payload)
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (input_json, predicted_price, user_id)
                VALUES (%s, %s, %s)
                """,
                (json.dumps(db_payload), round(float(predicted_price), 2), user_id),
            )
    finally:
        conn.close()


def get_all_listings(limit=500):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM property_listings
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
    finally:
        conn.close()
    return rows


def generate_recommendations(input_payload, predicted_price, top_n=6):
    listings = get_all_listings(limit=1000)
    if not listings:
        return []

    def numeric_similarity(a, b, min_baseline=1.0):
        av = abs(float(a))
        bv = abs(float(b))
        baseline = max(av, bv, float(min_baseline))
        diff = abs(float(a) - float(b))
        return max(0.0, 1.0 - (diff / baseline))

    target_state = _safe_str(input_payload.get("state") or input_payload.get("negeri"), "").lower()
    target_area_text = _safe_str(input_payload.get("area_text"), "").lower()
    target_property_type = _safe_int(input_payload.get("Property_Type"), 0)
    target_tenure = _safe_int(input_payload.get("Tenure"), 0)
    target_unit_type = _safe_int(input_payload.get("Unit_Type"), 0)
    target_bedroom = _safe_float(input_payload.get("Bedroom"), 0.0)
    target_bathroom = _safe_float(input_payload.get("Bathroom"), 0.0)
    target_car_park = _safe_float(input_payload.get("Car_Park"), 0.0)
    target_built_up = _safe_float(input_payload.get("Built_Up_SF"), 0.0)
    target_land_size = _safe_float(input_payload.get("Land_Size"), 0.0)

    ranked = []
    prediction_value = max(_safe_float(predicted_price, default=0.0), 1.0)
    reference_price = max(prediction_value, 1.0)
    for listing in listings:
        listing_state = _safe_str(listing.get("negeri"), "").lower()
        listing_area = _safe_str(listing.get("area"), "").lower()
        listing_property_type = _safe_int(listing.get("property_type"), 0)
        listing_tenure = _safe_int(listing.get("tenure"), 0)
        listing_unit_type = _safe_int(listing.get("unit_type"), 0)
        listing_bedroom = _safe_float(listing.get("bedroom"), 0.0)
        listing_bathroom = _safe_float(listing.get("bathroom"), 0.0)
        listing_car_park = _safe_float(listing.get("car_park"), 0.0)
        listing_built_up = _safe_float(listing.get("built_up_sf"), 0.0)
        listing_land_size = _safe_float(listing.get("land_size"), 0.0)
        listing_price = _safe_float(listing.get("listing_price"), default=0.0)
        if listing_price <= 0:
            continue

        state_match = 1.0 if target_state and listing_state == target_state else 0.0
        property_type_match = 1.0 if target_property_type and listing_property_type == target_property_type else 0.0
        tenure_match = 1.0 if target_tenure and listing_tenure == target_tenure else 0.0
        unit_type_match = 1.0 if target_unit_type and listing_unit_type == target_unit_type else 0.0
        area_match = 1.0 if target_area_text and target_area_text in listing_area else 0.0

        bedroom_similarity = numeric_similarity(target_bedroom, listing_bedroom, min_baseline=2.0)
        bathroom_similarity = numeric_similarity(target_bathroom, listing_bathroom, min_baseline=2.0)
        car_park_similarity = numeric_similarity(target_car_park, listing_car_park, min_baseline=2.0)
        built_up_similarity = numeric_similarity(target_built_up, listing_built_up, min_baseline=300.0)
        land_size_similarity = numeric_similarity(target_land_size, listing_land_size, min_baseline=300.0)

        non_price_factor_score = (
            (0.20 * state_match)
            + (0.15 * property_type_match)
            + (0.07 * tenure_match)
            + (0.06 * unit_type_match)
            + (0.10 * bathroom_similarity)
            + (0.05 * car_park_similarity)
            + (0.10 * built_up_similarity)
            + (0.07 * land_size_similarity)
            + (0.05 * area_match)
        )
        non_price_factor_score = min(max(non_price_factor_score, 0.0), 1.0)

        price_gap = abs(listing_price - prediction_value)
        price_gap_percent = (price_gap / reference_price) * 100.0
        price_similarity = max(0.0, 1.0 - (price_gap_percent / 100.0))
        bedroom_gap = abs(listing_bedroom - target_bedroom)
        bathroom_gap = abs(listing_bathroom - target_bathroom)
        final_score = (
            (0.50 * price_similarity)
            + (0.25 * bedroom_similarity)
            + (0.25 * non_price_factor_score)
        )
        final_score = min(max(final_score, 0.0), 1.0)

        # Bucket priority: keep recommendations close to prediction price and room count first.
        if state_match and price_gap_percent <= 30.0 and bedroom_gap <= 1.0:
            priority_bucket = 0
        elif state_match and price_gap_percent <= 45.0 and bedroom_gap <= 2.0:
            priority_bucket = 1
        elif price_gap_percent <= 45.0 and bedroom_gap <= 1.0:
            priority_bucket = 2
        elif state_match and price_gap_percent <= 60.0:
            priority_bucket = 3
        else:
            priority_bucket = 4

        listing["listing_price"] = listing_price
        listing["sql_listing_price"] = listing_price
        listing["price_gap"] = round(price_gap, 2)
        listing["price_gap_percent"] = round(price_gap_percent, 1)
        listing["price_similarity"] = round(price_similarity, 4)
        listing["bedroom_gap"] = round(bedroom_gap, 2)
        listing["bathroom_gap"] = round(bathroom_gap, 2)
        listing["factor_score"] = round(non_price_factor_score, 4)
        listing["final_score"] = final_score
        listing["score_percent"] = round(final_score * 100, 1)
        listing["priority_bucket"] = priority_bucket
        ranked.append(listing)

    ranked.sort(
        key=lambda item: (
            item["priority_bucket"],    # 1) strong price+room matches first
            -item["final_score"],       # 2) highest weighted similarity
            item["price_gap"],          # 3) closest listing_price
            item["bedroom_gap"],        # 4) closest bedroom count
        )
    )
    return ranked[:top_n]


def search_listings(filters, page=1, per_page=50):
    where = []
    params = []
    if filters.get("min_price") is not None:
        where.append("listing_price >= %s")
        params.append(filters["min_price"])
    if filters.get("max_price") is not None:
        where.append("listing_price <= %s")
        params.append(filters["max_price"])
    if filters.get("state"):
        where.append("negeri = %s")
        params.append(filters["state"])
    if filters.get("property_type") is not None:
        where.append("property_type = %s")
        params.append(filters["property_type"])
    if filters.get("bedroom") is not None:
        where.append("bedroom >= %s")
        params.append(filters["bedroom"])

    base_sql = "FROM property_listings"
    if where:
        base_sql += " WHERE " + " AND ".join(where)

    count_sql = "SELECT COUNT(*) AS total " + base_sql

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(count_sql, params)
            count_row = cursor.fetchone() or {}
            total_count = max(0, _safe_int(count_row.get("total"), 0))

            safe_per_page = max(1, min(500, _safe_int(per_page, 50)))
            total_pages = max(1, (total_count + safe_per_page - 1) // safe_per_page)
            safe_page = max(1, _safe_int(page, 1))
            if safe_page > total_pages:
                safe_page = total_pages

            offset = (safe_page - 1) * safe_per_page
            data_sql = "SELECT * " + base_sql + " ORDER BY listing_price ASC LIMIT %s OFFSET %s"
            data_params = list(params) + [safe_per_page, offset]
            cursor.execute(data_sql, data_params)
            rows = [normalize_property_row(row) for row in cursor.fetchall()]
    finally:
        conn.close()
    return {
        "rows": rows,
        "total_count": total_count,
        "page": safe_page,
        "per_page": safe_per_page,
        "total_pages": total_pages,
    }


def enrich_property_rows_for_display(rows):
    enriched = []
    for row in rows:
        item = normalize_property_row(row)
        property_id = _safe_int(item.get("id"), default=0)
        raw_image_url = _safe_str(item.get("image_url"), "").strip()
        item["image_url"] = _resolve_property_image_url_for_display(property_id, raw_image_url)
        enriched.append(item)
    return enriched


def _serialize_api_value(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, bytes):
        return None
    if isinstance(value, dict):
        return {str(key): _serialize_api_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_api_value(item) for item in value]
    return value


def _api_success(data=None, message="Success", status=200):
    payload = {"success": True, "message": message}
    if data is not None:
        payload["data"] = _serialize_api_value(data)
    return jsonify(payload), status


def _api_error(message, status=400, details=None):
    payload = {"success": False, "message": message}
    if details is not None:
        payload["details"] = _serialize_api_value(details)
    return jsonify(payload), status


def _serialize_property_for_api(row, include_description=False):
    if not row:
        return None

    property_id = _safe_int(row.get("id"), default=0)
    property_type_code = _safe_int(row.get("property_type"), default=0)
    furnishing_code = _safe_int(row.get("furnishing"), default=0)
    tenure_code = _safe_int(row.get("tenure"), default=0)
    unit_type_code = _safe_int(row.get("unit_type"), default=0)
    raw_image_url = _safe_str(row.get("image_url"), "").strip()

    payload = {
        "property_id": property_id,
        "property_code": f"P{property_id}" if property_id > 0 else None,
        "title": _safe_str(row.get("title")),
        "area": _safe_str(row.get("area")),
        "state": _safe_str(row.get("negeri")),
        "property_type_code": property_type_code,
        "property_type": PROPERTY_TYPE_OPTIONS.get(property_type_code, "Unknown"),
        "built_up_sf": _safe_float(row.get("built_up_sf"), default=0.0),
        "land_size": _safe_float(row.get("land_size"), default=0.0),
        "bedroom": _safe_int(row.get("bedroom"), default=0),
        "bathroom": _safe_int(row.get("bathroom"), default=0),
        "car_park": _safe_int(row.get("car_park"), default=0),
        "furnishing_code": furnishing_code,
        "furnishing": FURNISHING_OPTIONS.get(furnishing_code, "Unknown"),
        "tenure_code": tenure_code,
        "tenure": TENURE_OPTIONS.get(tenure_code, "Unknown"),
        "unit_type_code": unit_type_code,
        "unit_type": UNIT_TYPE_OPTIONS.get(unit_type_code, "Unknown"),
        "listing_price": _safe_float(row.get("listing_price"), default=0.0),
        "latitude": _serialize_api_value(row.get("latitude")),
        "longitude": _serialize_api_value(row.get("longitude")),
        "image_url": _resolve_property_image_url_for_display(property_id, raw_image_url),
        "created_at": _serialize_api_value(row.get("created_at")),
    }
    if include_description:
        payload["description"] = _safe_str(row.get("description"))

    for extra_key in (
        "final_score",
        "score_percent",
        "price_gap",
        "price_gap_percent",
        "bedroom_gap",
        "bathroom_gap",
        "sql_listing_price",
    ):
        if extra_key in row:
            payload[extra_key] = _serialize_api_value(row.get(extra_key))
    return payload


def _serialize_nearest_place_for_api(place_row):
    if not place_row:
        return None
    return {
        "name": _safe_str(place_row.get("name")),
        "distance_km": round(_safe_float(place_row.get("distance_km"), default=0.0), 3),
        "latitude": _serialize_api_value(place_row.get("latitude")),
        "longitude": _serialize_api_value(place_row.get("longitude")),
    }


def _parse_api_search_filters():
    filters = {
        "min_price": None,
        "max_price": None,
        "state": None,
        "property_type": None,
        "bedroom": None,
    }
    try:
        filters["min_price"] = parse_optional_filter_float(request.args, "min_price")
        filters["max_price"] = parse_optional_filter_float(request.args, "max_price")
        filters["state"] = request.args.get("state", "").strip() or None
        filters["property_type"] = parse_optional_filter_int(request.args, "property_type")
        filters["bedroom"] = parse_optional_filter_int(request.args, "bedroom")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return filters


def get_home_property_suggestions(user, limit=6):
    if not user:
        return {
            "state": None,
            "properties": [],
            "total_count": 0,
            "message": None,
        }

    state = _safe_str(user.get("state"), "")
    if not state:
        return {
            "state": None,
            "properties": [],
            "total_count": 0,
            "message": "Add your state in Profile to get property suggestions on your home page.",
        }

    filters = {"state": state}
    family_count = user.get("family_count")
    if family_count is not None:
        try:
            filters["bedroom"] = max(1, int(family_count))
        except (TypeError, ValueError):
            pass

    result = search_listings(filters, page=1, per_page=max(1, min(12, int(limit))))
    properties = enrich_property_rows_for_display(result["rows"])
    return {
        "state": state,
        "properties": properties,
        "total_count": result["total_count"],
        "message": None,
    }


def get_state_sales_distribution():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT negeri AS state_name, COUNT(*) AS total_listings
                FROM property_listings
                WHERE negeri IS NOT NULL AND negeri <> ''
                GROUP BY negeri
                ORDER BY total_listings DESC, negeri ASC
                """
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    labels = [str(row["state_name"]) for row in rows]
    counts = [int(row["total_listings"]) for row in rows]
    top_state = rows[0] if rows else None
    return {"labels": labels, "counts": counts, "top_state": top_state}


def get_state_highest_price_distribution():
    try:
        model = load_model()
        feature_columns = load_feature_columns()
    except Exception:
        return {"labels": [], "max_prices": [], "top_state": None}
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT negeri, built_up_sf, bathroom, furnishing, bedroom, tenure,
                       car_park, property_type, land_size, unit_type
                FROM property_listings
                WHERE negeri IS NOT NULL AND negeri <> ''
                """
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return {"labels": [], "max_prices": [], "top_state": None}

    work = pd.DataFrame(rows)
    work["negeri"] = work["negeri"].astype(str).str.strip()
    work = work[work["negeri"] != ""]
    if work.empty:
        return {"labels": [], "max_prices": [], "top_state": None}

    db_to_model_column_map = {
        "built_up_sf": "Built_Up_SF",
        "bathroom": "Bathroom",
        "furnishing": "Furnishing",
        "bedroom": "Bedroom",
        "tenure": "Tenure",
        "car_park": "Car_Park",
        "property_type": "Property_Type",
        "land_size": "Land_Size",
        "unit_type": "Unit_Type",
    }

    x_df = pd.DataFrame(0.0, index=work.index, columns=feature_columns, dtype=float)
    for db_col, model_col in db_to_model_column_map.items():
        if model_col in x_df.columns and db_col in work.columns:
            x_df[model_col] = pd.to_numeric(work[db_col], errors="coerce").fillna(0.0)

    for col in feature_columns:
        if col.startswith("negeri_"):
            state_name = col.replace("negeri_", "", 1)
            x_df[col] = (work["negeri"] == state_name).astype(float)

    y_pred = model.predict(x_df)
    work["predicted_price"] = pd.to_numeric(y_pred, errors="coerce")
    work = work.dropna(subset=["predicted_price"])
    if work.empty:
        return {"labels": [], "max_prices": [], "top_state": None}

    summary = (
        work.groupby("negeri", as_index=False)["predicted_price"]
        .max()
        .rename(columns={"negeri": "state_name", "predicted_price": "max_price"})
        .sort_values(by=["max_price", "state_name"], ascending=[False, True])
    )

    labels = summary["state_name"].astype(str).tolist()
    max_prices = summary["max_price"].astype(float).tolist()
    top_state = None
    if not summary.empty:
        top_state = {
            "state_name": str(summary.iloc[0]["state_name"]),
            "max_price": float(summary.iloc[0]["max_price"]),
        }
    return {"labels": labels, "max_prices": max_prices, "top_state": top_state}


def _load_model_dataset_for_training():
    candidate_paths = []
    local_dataset_candidates = [
        os.path.join(BASE_DIR, "dataset_with_url.csv"),
        os.path.join(BASE_DIR, "dataset_with_negeri_filled_with_src.csv"),
    ]
    # Prefer datasets in current project folder first, so notebook/system stay aligned.
    for path in local_dataset_candidates + [PROPERTY_SRC_DATASET_PATH, PROPERTY_DATASET_PATH]:
        clean = _safe_str(path, "")
        if clean and clean not in candidate_paths:
            candidate_paths.append(clean)

    for path in candidate_paths:
        loaded = _read_csv_safely(path)
        if loaded.empty:
            continue
        if "Price" not in set(loaded.columns):
            continue
        state_column = None
        if "negeri" in set(loaded.columns):
            state_column = "negeri"
        elif "Negeri" in set(loaded.columns):
            state_column = "Negeri"
        if state_column:
            return loaded, state_column, path
    return pd.DataFrame(), None, ""


def _build_model_feature_frame(work_df, state_column, feature_columns):
    work = work_df.copy()
    work["Price"] = pd.to_numeric(work["Price"], errors="coerce")
    for col in MODEL_NUMERIC_COLUMNS:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    work["state_text"] = work[state_column].astype(str).str.strip()
    work = work.dropna(subset=["Price"])
    if work.empty:
        return work, pd.DataFrame(), pd.Series(dtype=float)

    x_df = pd.DataFrame(0.0, index=work.index, columns=feature_columns, dtype=float)
    required_numeric = [col for col in MODEL_NUMERIC_COLUMNS if col in feature_columns]
    for col in required_numeric:
        x_df[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    for col in feature_columns:
        if col.startswith("negeri_"):
            state_name = col.replace("negeri_", "", 1)
            x_df[col] = (work["state_text"] == state_name).astype(float)
    y_series = work["Price"].astype(float)
    return work, x_df, y_series


@lru_cache(maxsize=1)
def get_model_quality_metrics():
    df, state_column, _ = _load_model_dataset_for_training()

    if df.empty or not state_column:
        return None

    model = load_model()
    feature_columns = load_feature_columns()
    work, x_df, y_true_all = _build_model_feature_frame(df, state_column, feature_columns)
    if len(y_true_all) < 10:
        return None

    _, test_index = train_test_split(
        work.index.to_numpy(),
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    x_test = x_df.loc[test_index]
    y_true = y_true_all.loc[test_index].astype(float).to_numpy()
    y_pred = model.predict(x_test)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    safe_y = np.where(np.asarray(y_true) <= 0, 1.0, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / safe_y)) * 100.0)
    estimated_accuracy = max(0.0, min(100.0, 100.0 - mape))

    if r2 >= 0.85:
        reliability = "High reliability"
    elif r2 >= 0.7:
        reliability = "Moderate reliability"
    else:
        reliability = "Use with caution"

    return {
        "sample_size": int(len(y_true)),
        "estimated_accuracy_pct": round(estimated_accuracy, 2),
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "reliability": reliability,
        "evaluation_note": (
            "Evaluated on a hold-out test split (20%) with random_state=42 by comparing predicted prices "
            "against actual listing prices from the selected dataset."
        ),
    }


def train_xgboost_after_tuned_model(test_size=0.2, random_state=42):
    df, state_column, source_path = _load_model_dataset_for_training()
    if df.empty or not state_column:
        raise ValueError("Unable to load training dataset with required columns (Price + negeri/Negeri).")

    work = df.copy()
    work["state_text"] = work[state_column].astype(str).str.strip()
    unique_states = sorted({s for s in work["state_text"].tolist() if _safe_str(s, "")})
    feature_columns = list(MODEL_NUMERIC_COLUMNS) + [f"negeri_{state}" for state in unique_states]

    _, x_df, y_series = _build_model_feature_frame(work, state_column, feature_columns)
    if len(y_series) < 20:
        raise ValueError("Dataset is too small for reliable training/testing split.")

    x_train, x_test, y_train, y_test = train_test_split(
        x_df,
        y_series,
        test_size=float(test_size),
        random_state=int(random_state),
        shuffle=True,
    )

    model = XGBRegressor(**XGBOOST_AFTER_TUNED_PARAMS)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    safe_y = np.where(np.asarray(y_test) <= 0, 1.0, y_test)
    mape = float(np.mean(np.abs((np.asarray(y_test) - np.asarray(y_pred)) / safe_y)) * 100.0)
    estimated_accuracy = max(0.0, min(100.0, 100.0 - mape))

    model_path = _resolve_project_path(os.getenv("MODEL_PATH", "models/best_model.pkl"))
    columns_path = _resolve_project_path(os.getenv("FEATURE_COLUMNS_PATH", "models/model_columns.pkl"))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(columns_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(feature_columns, columns_path)

    load_model.cache_clear()
    load_feature_columns.cache_clear()
    get_state_choices.cache_clear()
    get_model_quality_metrics.cache_clear()

    return {
        "model_path": model_path,
        "feature_columns_path": columns_path,
        "dataset_path": source_path,
        "sample_size": int(len(y_test)),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "estimated_accuracy_pct": round(estimated_accuracy, 2),
    }


@lru_cache(maxsize=2000)
def get_location_market_context(state_name, area_name):
    state_clean = _safe_str(state_name, "")
    area_clean = _safe_str(area_name, "")
    normalized_text = f"{area_clean} {state_clean}".lower().strip()

    mature_city_keywords = {
        "kuala lumpur",
        "kl",
        "petaling jaya",
        "pj",
        "subang",
        "shah alam",
        "cyberjaya",
        "putrajaya",
        "johor bahru",
        "george town",
        "pulau pinang",
        "penang",
    }
    developing_city_keywords = {
        "seremban",
        "nilai",
        "bangi",
        "kajang",
        "puchong",
        "klang",
        "rawang",
        "bukit mertajam",
        "pasir gudang",
        "kuantan",
        "ipoh",
        "melaka",
    }

    if any(keyword in normalized_text for keyword in mature_city_keywords):
        development_level = "mature city"
        development_note = "offers stronger access to daily amenities, transport links, and established economic activity"
        lifestyle_focus = "works well for buyers who prefer an active lifestyle and shorter travel time"
    elif any(keyword in normalized_text for keyword in developing_city_keywords):
        development_level = "developing city"
        development_note = "is expanding with growing residential and commercial projects"
        lifestyle_focus = "works well for buyers seeking growth potential over the medium term"
    else:
        development_level = "emerging area"
        development_note = "is more community-oriented, with amenities improving progressively"
        lifestyle_focus = "works well for buyers who value a calmer environment and neighborhood stability"

    return {
        "state_name": state_clean or "Unknown",
        "area_name": area_clean or "Unknown area",
        "development_level": development_level,
        "development_note": development_note,
        "lifestyle_focus": lifestyle_focus,
        # Keep legacy keys for compatibility with existing call sites.
        "state_listing_count": 0,
        "area_listing_count": 0,
        "state_avg_price": 0.0,
        "area_avg_price": 0.0,
        "area_avg_price_psf": 0.0,
        "area_vs_state_pct": 0.0,
        "market_heat": development_level,
        "location_positioning": development_note,
    }


def _generate_rule_based_property_match(row, location_context=None):
    property_type_id = _safe_int(row.get("property_type"), default=0)
    property_type_name = PROPERTY_TYPE_OPTIONS.get(property_type_id, "Unknown")
    bedroom = max(0, _safe_int(row.get("bedroom"), default=0))
    bathroom = max(0, _safe_int(row.get("bathroom"), default=0))
    car_park = max(0, _safe_int(row.get("car_park"), default=0))
    built_up = max(0.0, _safe_float(row.get("built_up_sf"), default=0.0))
    state_name = _safe_str(row.get("negeri"), "Unknown")
    area_name = _safe_str(row.get("area"), "Unknown area")
    location_context = location_context or get_location_market_context(state_name, area_name)
    development_level = _safe_str(location_context.get("development_level"), "emerging area")
    development_note = _safe_str(
        location_context.get("development_note"),
        "is more community-oriented, with amenities improving progressively",
    )
    lifestyle_focus = _safe_str(
        location_context.get("lifestyle_focus"),
        "works well for buyers who value a balanced lifestyle between convenience and calm living",
    )

    is_landed = property_type_id in {1, 2, 5, 6, 7, 9}
    is_highrise = property_type_id in {3, 4, 8, 10}

    if bedroom <= 2:
        suitable_for = "Single occupant / newly married couple"
    elif bedroom == 3:
        suitable_for = "Small family (3-4 members)"
    elif bedroom == 4:
        suitable_for = "Medium-large family (4-6 members)"
    else:
        suitable_for = "Large or multi-generational family (6+ members)"

    score = 55
    if is_landed:
        score += 15
    if bedroom >= 4:
        score += 15
    if bathroom >= 3:
        score += 7
    if car_park >= 2:
        score += 6
    if built_up >= 1400:
        score += 7
    if is_highrise and bedroom <= 3:
        score += 6
    if development_level == "mature city":
        score += 5
    elif development_level == "developing city":
        score += 4
    if bathroom >= max(2, bedroom // 2):
        score += 4
    if car_park >= 2 and bedroom >= 3:
        score += 3
    score = max(40, min(98, score))

    if score >= 82:
        match_level = "Strong Match"
    elif score >= 68:
        match_level = "Good Match"
    else:
        match_level = "Consider"

    summary = (
        f"{area_name}, {state_name} is categorized as a {development_level}; the area {development_note}. "
        f"For a {property_type_name} with {bedroom} bedrooms and {bathroom} bathrooms, it is suitable for {suitable_for.lower()}."
    )

    suggestions = []
    if development_level == "mature city":
        suggestions.append(
            f"{area_name} is a stronger fit if you want quick access to city facilities, jobs, and transport."
        )
    elif development_level == "developing city":
        suggestions.append(
            f"{area_name} is a developing urban area, suitable for buyers who prefer neighborhoods with growth momentum."
        )
    else:
        suggestions.append(
            f"{area_name} is typically calmer and more stable, suitable for buyers prioritizing community atmosphere over city intensity."
        )

    if bedroom >= 4:
        suggestions.append(
            "A higher bedroom count is suitable for larger families, multi-generational households, or home-office needs."
        )
    elif bedroom == 3:
        suggestions.append(
            "3 bedrooms are usually a balanced option for small to medium families."
        )
    else:
        suggestions.append(
            "A smaller bedroom count is suitable for individuals or couples who prefer lower maintenance needs."
        )

    if bathroom < max(2, bedroom // 2):
        suggestions.append(
            "Ensure the bathroom-to-bedroom ratio is practical enough for day-to-day household comfort."
        )
    else:
        suggestions.append(
            f"With the current room profile, this unit {lifestyle_focus}."
        )

    return {
        "suitable_for": suitable_for,
        "match_level": match_level,
        "score": score,
        "summary": summary,
        "suggestions": suggestions[:3],
        "source": "fallback",
    }


@lru_cache(maxsize=1500)
def _generate_ai_property_match_llm_cached(signature_json):
    payload = json.loads(signature_json)
    style_modes = ["family-practical", "urban-lifestyle", "budget-conscious", "space-planning", "investment-minded"]
    style_hint = style_modes[_safe_int(payload.get("style_seed"), 0) % len(style_modes)]
    temp = _safe_float(os.getenv("LLM_MATCH_TEMPERATURE", "0.95"), 0.95)

    system_prompt = (
        "You are an LLM-powered Property Assistant for Malaysian real-estate search results. "
        "Return STRICT JSON only with keys: suitable_for, match_level, score, summary, suggestions. "
        "suggestions must be an array of exactly 3 concise strings. "
        "Use match_level only from: Strong Match, Good Match, Consider. "
        "Avoid starting summary with 'This property' or 'This <type>'. "
        "Do not mention dataset counts, averages, percentages, or internal database metrics. "
        "Focus on general location suitability: whether the area is a mature city, a developing city, or an emerging area, plus room suitability."
    )
    user_prompt = (
        f"Create a unique recommendation style: {style_hint}.\n"
        "Avoid repetitive sentence openings and avoid generic copy-paste wording.\n\n"
        "Property profile:\n"
        f"- Property ID: {payload.get('property_id')}\n"
        f"- Type: {payload.get('property_type_name')}\n"
        f"- State: {payload.get('state')}\n"
        f"- Area: {payload.get('area')}\n"
        f"- Development Level: {payload.get('development_level')}\n"
        f"- Development Note: {payload.get('development_note')}\n"
        f"- Lifestyle Focus: {payload.get('lifestyle_focus')}\n"
        f"- Bedroom: {payload.get('bedroom')}\n"
        f"- Bathroom: {payload.get('bathroom')}\n"
        f"- Car Park: {payload.get('car_park')}\n"
        f"- Built-up (sqft): {payload.get('built_up_sf')}\n\n"
        "Output JSON only. At least 2 out of 3 suggestions must be location-focused and 1 suggestion must mention bedroom suitability."
    )
    return _call_llm_chat_json(system_prompt, user_prompt, temperature=temp, max_tokens=300)


def _sanitize_ai_match_response(candidate, fallback):
    if not isinstance(candidate, dict):
        return fallback

    suitable_for = _safe_str(candidate.get("suitable_for"), fallback["suitable_for"])
    summary = _safe_str(candidate.get("summary"), fallback["summary"])
    score = max(35, min(99, _safe_int(candidate.get("score"), fallback["score"])))

    allowed_levels = {"Strong Match", "Good Match", "Consider"}
    match_level = _safe_str(candidate.get("match_level"), "")
    if match_level not in allowed_levels:
        if score >= 82:
            match_level = "Strong Match"
        elif score >= 68:
            match_level = "Good Match"
        else:
            match_level = "Consider"

    llm_suggestions = candidate.get("suggestions")
    suggestions = []
    seen = set()
    if isinstance(llm_suggestions, list):
        for tip in llm_suggestions:
            clean_tip = _safe_str(tip, "")
            if not clean_tip:
                continue
            key = clean_tip.lower()
            if key in seen:
                continue
            suggestions.append(clean_tip)
            seen.add(key)
            if len(suggestions) >= 3:
                break

    generic_topups = [
        "Review commute convenience, neighborhood activity, and day-to-day accessibility before finalizing.",
        "Match the bedroom count with your current household size and expected 3-5 year needs.",
        "Compare nearby options in the same area to validate whether this location fits your lifestyle goals.",
    ]
    for topup in generic_topups:
        key = topup.lower()
        if key in seen:
            continue
        suggestions.append(topup)
        seen.add(key)
        if len(suggestions) >= 3:
            break

    return {
        "suitable_for": suitable_for,
        "match_level": match_level,
        "score": score,
        "summary": summary,
        "suggestions": suggestions[:3],
        "source": "llm",
    }


def _build_llm_unavailable_match(fallback, area_name, state_name, error_detail=""):
    return {
        "suitable_for": fallback.get("suitable_for", "General household fit"),
        "match_level": "Consider",
        "score": fallback.get("score", 60),
        "summary": (
            f"AI generation is temporarily unavailable for {area_name}, {state_name}. "
            "The recommendation cannot be generated from the API right now."
        ),
        "suggestions": [
            "Refresh the search to trigger a fresh AI generation attempt.",
            "Check your API key/model configuration if this issue continues.",
            "Use the property details page while AI generation recovers.",
        ],
        "source": "llm_error",
        "error_detail": _safe_str(error_detail, ""),
    }


def generate_ai_property_match(
    row,
    use_llm=True,
    row_rank=0,
    search_nonce=0,
    variation_seed=0,
    location_context=None,
    allow_fallback=False,
):
    state_name = _safe_str(row.get("negeri"), "Unknown")
    area_name = _safe_str(row.get("area"), "Unknown area")
    location_context = location_context or get_location_market_context(state_name, area_name)
    fallback = _generate_rule_based_property_match(row, location_context=location_context)
    if (not use_llm) or (not _llm_is_configured()):
        return (
            fallback
            if allow_fallback
            else _build_llm_unavailable_match(
                fallback,
                area_name,
                state_name,
                error_detail="LLM API is not configured.",
            )
        )

    property_id = _safe_int(row.get("id"), default=0)
    property_type_id = _safe_int(row.get("property_type"), default=0)
    signature_payload = {
        "property_id": property_id,
        "property_type_name": PROPERTY_TYPE_OPTIONS.get(property_type_id, "Unknown"),
        "state": state_name,
        "area": area_name,
        "development_level": _safe_str(location_context.get("development_level"), "emerging area"),
        "development_note": _safe_str(
            location_context.get("development_note"),
            "is more community-oriented, with amenities improving progressively",
        ),
        "lifestyle_focus": _safe_str(
            location_context.get("lifestyle_focus"),
            "works well for buyers who value a balanced lifestyle between convenience and calm living",
        ),
        "bedroom": _safe_int(row.get("bedroom"), 0),
        "bathroom": _safe_int(row.get("bathroom"), 0),
        "car_park": _safe_int(row.get("car_park"), 0),
        "built_up_sf": round(_safe_float(row.get("built_up_sf"), 0.0), 1),
        "style_seed": (
            (property_id if property_id > 0 else _safe_int(row.get("bedroom"), 0)) * 37
            + (_safe_int(row_rank, 0) * 17)
            + (_safe_int(search_nonce, 0) % 997)
            + (_safe_int(variation_seed, 0) * 29)
        ),
        "row_rank": _safe_int(row_rank, 0),
        "variation_seed": _safe_int(variation_seed, 0),
    }
    signature_json = json.dumps(signature_payload, sort_keys=True)
    try:
        llm_result = _generate_ai_property_match_llm_cached(signature_json)
        return _sanitize_ai_match_response(llm_result, fallback)
    except Exception as exc:
        if allow_fallback:
            return fallback
        error_message = _safe_str(str(exc), "").strip()
        if not error_message:
            error_message = repr(exc)
        return _build_llm_unavailable_match(
            fallback,
            area_name,
            state_name,
            error_detail=error_message,
        )


@lru_cache(maxsize=500)
def _generate_family_profile_insight_llm_cached(signature_json):
    payload = json.loads(signature_json)
    temp = _safe_float(os.getenv("LLM_FAMILY_INSIGHT_TEMPERATURE", "0.55"), 0.55)
    system_prompt = (
        "You are the Smart Property Valuer Family Insight assistant. "
        "Return STRICT JSON only with one key: insight. "
        "insight must be a single concise English sentence (20-40 words). "
        "The sentence must describe the household profile and recommend suitable house type(s). "
        "Do not use fixed family-size rules from code; infer naturally from profile context and candidates. "
        "If assume_single is true, clearly mention that the system temporarily assumes one individual "
        "until profile details are completed."
    )
    user_prompt = (
        "Generate Family Profile Insight using this payload:\n"
        f"- Marital Status: {payload.get('marital_status')}\n"
        f"- Family Count: {payload.get('family_count')}\n"
        f"- Household Profile: {payload.get('household_profile')}\n"
        f"- Search Context Property Type: {payload.get('search_context_property_type')}\n"
        f"- Candidate Residential House Types: {payload.get('candidate_house_types')}\n"
        f"- Profile Complete: {payload.get('profile_complete')}\n"
        f"- Assume Single: {payload.get('assume_single')}\n"
        f"- Missing Marital Status: {payload.get('missing_marital_status')}\n"
        f"- Missing Family Count: {payload.get('missing_family_count')}\n"
    )
    return _call_llm_chat_json(system_prompt, user_prompt, temperature=temp, max_tokens=140)


def _sanitize_family_profile_insight(candidate, fallback_text):
    if not isinstance(candidate, dict):
        return fallback_text
    raw = _safe_str(candidate.get("insight"), "").strip()
    if not raw:
        raw = _safe_str(candidate.get("family_profile_insight"), "").strip()
    clean = " ".join(raw.split())
    if not clean:
        return fallback_text
    if len(clean) > 320:
        clean = clean[:317].rstrip(" ,;:") + "..."
    return clean


def _build_user_family_profile_prompt(user_profile, preferred_house_type=""):
    user_profile = user_profile or {}
    search_context_type = _safe_str(preferred_house_type, "").strip()
    if search_context_type.lower() == "unknown":
        search_context_type = ""

    missing_marital_status = False
    missing_family_count = False
    if not user_profile:
        family_count = 1
        missing_marital_status = True
        missing_family_count = True
        marital_label = "Not specified"
        family_label = "1 family member"
        household_desc = "the household profile is likely small"
        fallback_prompt = (
            "Marital Status or Number of Family is not filled in yet; Smart Property Valuer will assist with "
            "prediction support and the system will temporarily assume 1 individual, "
            "and the house-type recommendation will be generated by API once profile details are available."
        )
    else:
        marital_key = _safe_str(user_profile.get("marital_status"), "").strip().lower()
        marital_is_valid = marital_key in MARITAL_STATUS_OPTIONS
        marital_label = MARITAL_STATUS_OPTIONS.get(marital_key, "Not specified")
        raw_family_count = max(0, _safe_int(user_profile.get("family_count"), 0))
        family_count_assumed = raw_family_count <= 0
        family_count = 1 if family_count_assumed else raw_family_count
        missing_marital_status = not marital_is_valid
        missing_family_count = family_count_assumed

        if family_count == 1:
            family_label = "1 family member"
        else:
            family_label = f"{family_count} family members"

        if family_count <= 1:
            household_desc = "the household profile is likely small"
        elif family_count <= 3:
            household_desc = "the household profile is small to medium"
        elif family_count <= 5:
            household_desc = "the household profile is medium-sized"
        else:
            household_desc = "the household profile is large"

        if family_count_assumed or (not marital_is_valid):
            fallback_prompt = (
                "Marital Status or Number of Family is incomplete; Smart Property Valuer will assist with prediction support "
                "and the system currently assumes 1 individual until your profile is updated, "
                "and the house-type recommendation will be generated by API from your latest profile context."
            )
        else:
            fallback_prompt = (
                f"Based on your profile ({marital_label}, {family_label}), {household_desc} and you are more suitable "
                "for house types suggested by the API from your current family profile."
            )

    profile_complete = not (missing_marital_status or missing_family_count)
    if not _llm_is_configured():
        return fallback_prompt

    candidate_house_types = [
        PROPERTY_TYPE_OPTIONS.get(key)
        for key in sorted(PROPERTY_TYPE_OPTIONS.keys())
        if key in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    ]
    candidate_house_types = [item for item in candidate_house_types if _safe_str(item, "")]

    signature_payload = {
        "marital_status": marital_label,
        "family_count": family_count,
        "household_profile": household_desc,
        "search_context_property_type": search_context_type or "Not specified",
        "candidate_house_types": ", ".join(candidate_house_types),
        "profile_complete": profile_complete,
        "assume_single": not profile_complete,
        "missing_marital_status": missing_marital_status,
        "missing_family_count": missing_family_count,
    }
    signature_json = json.dumps(signature_payload, sort_keys=True)
    try:
        llm_result = _generate_family_profile_insight_llm_cached(signature_json)
        return _sanitize_family_profile_insight(llm_result, fallback_prompt)
    except Exception:
        return fallback_prompt


def build_search_ai_overview(rows, filters=None, user_profile=None):
    if not rows:
        return None

    filters = filters or {}

    def _match_score(row):
        ai_match = row.get("ai_match")
        if not isinstance(ai_match, dict):
            return 0
        return max(0, min(100, _safe_int(ai_match.get("score"), 0)))

    def _match_level(row):
        ai_match = row.get("ai_match")
        if not isinstance(ai_match, dict):
            return "Consider"
        return _safe_str(ai_match.get("match_level"), "Consider")

    def _suitable_for(row):
        ai_match = row.get("ai_match")
        if not isinstance(ai_match, dict):
            return "General household fit"
        return _safe_str(ai_match.get("suitable_for"), "General household fit")

    def _to_pick(row):
        property_type_code = _safe_int(row.get("property_type"), 0)
        return {
            "id": _safe_int(row.get("id"), 0),
            "title": _safe_str(row.get("title"), "Property"),
            "area": _safe_str(row.get("area"), "Unknown area"),
            "state": _safe_str(row.get("negeri"), "Unknown"),
            "property_type_name": PROPERTY_TYPE_OPTIONS.get(property_type_code, "Unknown"),
            "price": round(_safe_float(row.get("listing_price"), 0.0), 2),
            "bedroom": _safe_int(row.get("bedroom"), 0),
            "bathroom": _safe_int(row.get("bathroom"), 0),
            "score": _match_score(row),
            "match_level": _match_level(row),
            "suitable_for": _suitable_for(row),
        }

    top_match_row = max(
        rows,
        key=lambda row: (
            _match_score(row),
            -_safe_float(row.get("listing_price"), 0.0),
            _safe_int(row.get("id"), 0),
        ),
    )

    budget_pool = [row for row in rows if _match_score(row) >= 68]
    if not budget_pool:
        budget_pool = list(rows)
    budget_pick_row = min(
        budget_pool,
        key=lambda row: (
            _safe_float(row.get("listing_price"), float("inf")),
            -_match_score(row),
            _safe_int(row.get("id"), 0),
        ),
    )

    space_pick_row = max(
        rows,
        key=lambda row: (
            _safe_int(row.get("bedroom"), 0),
            _safe_float(row.get("built_up_sf"), 0.0),
            _match_score(row),
            -_safe_float(row.get("listing_price"), 0.0),
        ),
    )

    action_items = []
    if filters.get("max_price") is None:
        action_items.append("Set Max Price to narrow results to your budget limit.")
    if not filters.get("state"):
        action_items.append("Choose a State filter to get location-specific recommendations.")
    if filters.get("bedroom") is None:
        action_items.append("Set Min Bedroom to align results with your household needs.")
    if not action_items:
        action_items.append("Open Details for Top Match and compare nearby facilities before deciding.")
        action_items.append("Shortlist 2-3 listings, then compare tenure, room layout, and commute access.")

    top_match_pick = _to_pick(top_match_row)
    budget_pick = _to_pick(budget_pick_row)
    space_pick = _to_pick(space_pick_row)
    family_profile_prompt = _build_user_family_profile_prompt(
        user_profile,
        preferred_house_type=top_match_pick.get("property_type_name"),
    )

    return {
        "panel_note": (
            "Assistant actions below are generated from your current search filters and LLM match output "
            "to help you shortlist faster."
        ),
        "family_profile_prompt": family_profile_prompt,
        "top_match": top_match_pick,
        "budget_pick": budget_pick,
        "space_pick": space_pick,
        "action_items": action_items[:3],
    }


def parse_optional_filter_float(args, key):
    raw = args.get(key, "").strip()
    if raw == "":
        return None
    return float(raw)


def parse_optional_filter_int(args, key):
    raw = args.get(key, "").strip()
    if raw == "":
        return None
    return int(raw)


def parse_positive_int(args, key, default):
    raw = args.get(key, "").strip()
    if raw == "":
        return int(default)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _format_input_number(value, suffix=""):
    if value is None:
        return "-"
    number = _safe_float(value, None)
    if number is None:
        return "-"
    if float(number).is_integer():
        formatted = f"{int(number):,}"
    else:
        formatted = f"{number:,.2f}"
    return f"{formatted}{suffix}"


def format_prediction_input_for_display(input_payload):
    if not isinstance(input_payload, dict):
        return []

    state_text = _safe_str(input_payload.get("state") or input_payload.get("negeri"), "-")
    area_text = _safe_str(input_payload.get("area_text") or input_payload.get("area"), "-")

    furnishing_code = _safe_int(input_payload.get("Furnishing"), 0)
    tenure_code = _safe_int(input_payload.get("Tenure"), 0)
    property_type_code = _safe_int(input_payload.get("Property_Type"), 0)
    unit_type_code = _safe_int(input_payload.get("Unit_Type"), 0)

    display_items = [
        {"label": "State", "value": state_text},
        {"label": "Area", "value": area_text},
        {"label": "Built-up Size", "value": _format_input_number(input_payload.get("Built_Up_SF"), " sqft")},
        {"label": "Land Size", "value": _format_input_number(input_payload.get("Land_Size"), " sqft")},
        {"label": "Bedrooms", "value": _format_input_number(input_payload.get("Bedroom"))},
        {"label": "Bathrooms", "value": _format_input_number(input_payload.get("Bathroom"))},
        {"label": "Car Parks", "value": _format_input_number(input_payload.get("Car_Park"))},
        {"label": "Furnishing", "value": FURNISHING_OPTIONS.get(furnishing_code, "Unknown")},
        {"label": "Tenure", "value": TENURE_OPTIONS.get(tenure_code, "Unknown")},
        {"label": "Property Type", "value": PROPERTY_TYPE_OPTIONS.get(property_type_code, "Unknown")},
        {"label": "Unit Type", "value": UNIT_TYPE_OPTIONS.get(unit_type_code, "Unknown")},
    ]

    return [item for item in display_items if _safe_str(item["value"], "-") != "-"]


def extract_contact_number(description_text):
    raw = _safe_str(description_text, "")
    if not raw:
        return None

    pattern = r"(?:(?:\+?6?01)[0-46-9]-?[\s.-]?\d{7,8}|\(?01[0-46-9]\)?[\s.-]?\d{7,8})"
    match = re.search(pattern, raw)
    if not match:
        return None

    digits = re.sub(r"[^\d]", "", match.group(0))
    if digits.startswith("60"):
        digits = digits[2:]
    return digits or None


def get_property_by_id(property_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM property_listings WHERE id = %s", (property_id,))
            row = cursor.fetchone()
    finally:
        conn.close()
    return normalize_property_row(row) if row else None


def resolve_property_image_filename(property_id):
    image_dir = _safe_str(PROPERTY_IMAGE_DIR, "")
    if not image_dir or not os.path.isdir(image_dir):
        return None

    pid = _safe_int(property_id, default=0)
    if pid <= 0:
        return None

    base_candidates = [
        f"P{pid}",
        f"P{pid:02d}",
        f"P{pid:03d}",
        f"P{pid:04d}",
    ]
    extensions = [".jpg", ".jpeg", ".png", ".webp"]

    for base_name in base_candidates:
        for ext in extensions:
            file_name = f"{base_name}{ext}"
            if os.path.exists(os.path.join(image_dir, file_name)):
                return file_name
    return None


def resolve_remote_property_image_url(property_id):
    pid = _safe_int(property_id, default=0)
    if pid <= 0:
        return None
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT image_url
                FROM property_listings
                WHERE id = %s
                LIMIT 1
                """,
                (pid,),
            )
            row = cursor.fetchone()
        image_url = _safe_str((row or {}).get("image_url"), "").strip()
        return _resolve_property_image_url_for_display(pid, image_url)
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


def prediction_form_context(submitted_data=None):
    return {
        "state_choices": get_state_choices(),
        "furnishing_options": FURNISHING_OPTIONS,
        "tenure_options": TENURE_OPTIONS,
        "property_type_options": PROPERTY_TYPE_OPTIONS,
        "unit_type_options": UNIT_TYPE_OPTIONS,
        "submitted_data": submitted_data or {},
    }


def _api_database_status():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        return "connected"
    except Exception:
        return "unavailable"
    finally:
        if conn:
            conn.close()


@app.route("/api/health", methods=["GET"])
def api_health():
    return _api_success(
        {
            "service": "Smart Property Valuer API",
            "status": "online",
            "database": _api_database_status(),
            "model_loaded": os.path.exists(_resolve_project_path(os.getenv("MODEL_PATH", "models/best_model.pkl"))),
            "llm_configured": _llm_is_configured(),
        },
        message="API is running",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    ensure_database_tables()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _api_error("Request body must be JSON.", status=400)

    try:
        input_payload, feature_df = parse_prediction_form(payload)
        model = load_model()
        prediction = float(model.predict(feature_df)[0])
        recommendations = generate_recommendations(input_payload, prediction)
        enriched_recommendations = enrich_property_rows_for_display(recommendations)
        return _api_success(
            {
                "predicted_price": round(prediction, 2),
                "input": input_payload,
                "recommendations": [
                    _serialize_property_for_api(item) for item in enriched_recommendations
                ],
            },
            message="Prediction completed successfully",
        )
    except Exception as exc:
        return _api_error(str(exc), status=400)


@app.route("/api/search", methods=["GET"])
def api_search():
    ensure_database_tables()
    try:
        filters = _parse_api_search_filters()
        page = parse_positive_int(request.args, "page", 1)
        per_page = parse_positive_int(request.args, "per_page", 25)
    except ValueError as exc:
        return _api_error(str(exc), status=400)

    per_page = max(1, min(100, per_page))
    search_result = search_listings(filters, page=page, per_page=per_page)
    rows = enrich_property_rows_for_display(search_result["rows"])
    return _api_success(
        {
            "filters": filters,
            "pagination": {
                "page": search_result["page"],
                "per_page": search_result["per_page"],
                "total_pages": search_result["total_pages"],
                "total_count": search_result["total_count"],
            },
            "results": [_serialize_property_for_api(row) for row in rows],
        },
        message="Search completed successfully",
    )


@app.route("/api/property/<int:property_id>", methods=["GET"])
def api_property_detail(property_id):
    ensure_database_tables()
    row = get_property_by_id(property_id)
    if not row:
        return _api_error("Property not found.", status=404)

    nearest = {
        "hospital": None,
        "primary_school": None,
        "secondary_school": None,
    }
    if row.get("latitude") is not None and row.get("longitude") is not None:
        property_lat = float(row["latitude"])
        property_lon = float(row["longitude"])
        nearest["hospital"] = find_nearest_place(property_lat, property_lon, load_hospital_dataset())
        nearest["primary_school"] = find_nearest_place(
            property_lat, property_lon, load_primary_school_dataset()
        )
        nearest["secondary_school"] = find_nearest_place(
            property_lat, property_lon, load_secondary_school_dataset()
        )

    property_payload = _serialize_property_for_api(row, include_description=True)
    property_payload["contact_number"] = extract_contact_number(row.get("description"))
    property_payload["nearest_facilities"] = {
        key: _serialize_nearest_place_for_api(value) for key, value in nearest.items()
    }
    return _api_success(property_payload, message="Property details retrieved successfully")


@app.route("/api/model/metrics", methods=["GET"])
def api_model_metrics():
    ensure_database_tables()
    metrics = _get_admin_model_metrics()
    return _api_success(
        {
            "model_name": metrics.get("model_name"),
            "r2": metrics.get("r2"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "reliability": metrics.get("reliability"),
            "evaluation_note": metrics.get("evaluation_note"),
            "sample_size": metrics.get("sample_size"),
            "last_training_date": metrics.get("last_training_date"),
            "parameter_items": metrics.get("parameter_items"),
        },
        message="Model metrics retrieved successfully",
    )


@app.route("/", methods=["GET"])
def home():
    admin_redirect = _redirect_admin_to_dashboard()
    if admin_redirect:
        return admin_redirect
    ensure_database_tables()
    sales_distribution = get_state_sales_distribution()
    highest_price_distribution = get_state_highest_price_distribution()
    model_metrics = get_model_quality_metrics()
    if model_metrics is None:
        # Retry once in case an old cached miss was stored before data/model became available.
        get_model_quality_metrics.cache_clear()
        model_metrics = get_model_quality_metrics()

    user = get_current_user() if session.get("user_id") else None
    property_suggestions = get_home_property_suggestions(user)

    return render_template(
        "home.html",
        sales_distribution=sales_distribution,
        highest_price_distribution=highest_price_distribution,
        model_metrics=model_metrics,
        property_suggestions=property_suggestions,
        property_type_options=PROPERTY_TYPE_OPTIONS,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    admin_redirect = _redirect_admin_to_dashboard()
    if admin_redirect:
        return admin_redirect
    ensure_database_tables()
    if session.get("user_id"):
        return redirect(url_for("profile"))

    if request.method == "GET":
        return render_template("register.html")

    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not email or not password:
        flash("Email and password are required.", "danger")
        return render_template("register.html", form_data=request.form)
    if len(password) < 6:
        flash("Password must be at least 6 characters.", "danger")
        return render_template("register.html", form_data=request.form)
    if password != confirm_password:
        flash("Passwords do not match.", "danger")
        return render_template("register.html", form_data=request.form)
    if get_user_by_email(email):
        flash("An account with this email already exists.", "danger")
        return render_template("register.html", form_data=request.form)

    user_id = create_user(email, password, full_name, user_role="user")
    session.clear()
    session["user_id"] = user_id
    session["user_email"] = email
    flash("Account created successfully.", "success")
    return redirect(url_for("profile"))


@app.route("/login", methods=["GET", "POST"])
def login():
    ensure_database_tables()
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    if session.get("user_id"):
        return redirect(url_for("home"))

    raw_next = request.args.get("next") or request.form.get("next")
    selected_role = _safe_str(
        request.form.get("login_as") or request.args.get("login_as"),
        "user",
    ).strip().lower()
    if selected_role not in {"user", "admin"}:
        selected_role = "user"
    next_url = _admin_safe_next_url(raw_next) if selected_role == "admin" else _safe_next_url(raw_next)

    if request.method == "GET":
        return render_template("login.html", next_url=next_url, selected_role=selected_role)

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    user = get_user_by_email(email)

    if selected_role == "admin" and email == ADMIN_DEFAULT_EMAIL.lower() and password == ADMIN_DEFAULT_PASSWORD:
        session.clear()
        session["is_admin"] = True
        session["admin_email"] = email
        session["admin_name"] = ADMIN_DEFAULT_NAME
        flash("Admin login successful.", "success")
        return redirect(_admin_safe_next_url(raw_next))

    if not user or not check_password_hash(user["password_hash"], password):
        flash("Invalid email or password.", "danger")
        return render_template(
            "login.html",
            next_url=next_url,
            form_data=request.form,
            selected_role=selected_role,
        )

    account_role = _safe_str(user.get("user_role"), "user").strip().lower()
    if account_role not in {"user", "admin"}:
        account_role = "user"
    if selected_role != account_role:
        flash("Selected login status does not match your account role.", "danger")
        return render_template(
            "login.html",
            next_url=next_url,
            form_data=request.form,
            selected_role=selected_role,
        )

    session.clear()
    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    if account_role == "admin":
        session["is_admin"] = True
        session["admin_email"] = user["email"]
        session["admin_name"] = _safe_str(user.get("full_name"), "").strip() or ADMIN_DEFAULT_NAME
        flash("Welcome back, admin.", "success")
        return redirect(_admin_safe_next_url(raw_next))

    session.pop("is_admin", None)
    session.pop("admin_email", None)
    session.pop("admin_name", None)
    flash("Welcome back.", "success")
    return redirect(_safe_next_url(raw_next))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    ensure_database_tables()
    if request.method == "GET":
        return render_template("forgot_password.html")

    email = _safe_str(request.form.get("email"), "").strip().lower()
    reason = _safe_str(request.form.get("reason"), "").strip()
    if not email:
        flash("Email is required.", "danger")
        return render_template("forgot_password.html", form_data=request.form)

    request_id = create_password_reset_request(email, reason=reason)
    if request_id is None:
        flash("Email is not registered in the system.", "danger")
        return render_template("forgot_password.html", form_data=request.form)

    flash(
        "Your password reset request has been submitted successfully. Please wait for admin approval before setting a new password.",
        "success",
    )
    return redirect(url_for("password_reset_status", email=email))


@app.route("/password-reset-status", methods=["GET", "POST"])
def password_reset_status():
    ensure_database_tables()
    checked_email = _safe_str(request.args.get("email"), "").strip().lower()
    request_row = None
    status_message = None

    if request.method == "POST":
        checked_email = _safe_str(request.form.get("email"), "").strip().lower()

    if checked_email:
        request_row = get_latest_password_reset_request(checked_email)
        if request_row:
            status_value = _normalize_password_reset_status(request_row.get("status"))
            request_row["status"] = status_value
            if status_value == "Pending":
                status_message = "Your password reset request is still pending. Please wait for admin approval."
            elif status_value == "Approved":
                status_message = "Your password reset request has been approved. You may now create a new password."
            elif status_value == "Rejected":
                status_message = (
                    "Your password reset request has been rejected. "
                    "Please contact the system administrator for further assistance."
                )
            elif status_value == "Completed":
                status_message = "Your password has already been reset successfully."
        elif request.method == "POST":
            flash("No password reset request found for this email.", "warning")

    return render_template(
        "password_reset_status.html",
        checked_email=checked_email,
        request_row=request_row,
        status_message=status_message,
    )


@app.route("/set-new-password", methods=["GET", "POST"])
def set_new_password():
    ensure_database_tables()
    email = _safe_str(request.args.get("email") or request.form.get("email"), "").strip().lower()
    if not email:
        flash("Email is required to reset password.", "warning")
        return redirect(url_for("password_reset_status"))

    latest_request = get_latest_password_reset_request(email)
    if not latest_request:
        flash("No password reset request found for this email.", "warning")
        return redirect(url_for("password_reset_status", email=email))

    latest_status = _normalize_password_reset_status(latest_request.get("status"))
    if latest_status != "Approved":
        flash("You can only set a new password after admin approval.", "warning")
        return redirect(url_for("password_reset_status", email=email))

    if request.method == "GET":
        return render_template("set_new_password.html", email=email)

    new_password = _safe_str(request.form.get("new_password"), "")
    confirm_password = _safe_str(request.form.get("confirm_password"), "")
    if not new_password:
        flash("Password cannot be empty.", "danger")
        return render_template("set_new_password.html", email=email)
    if len(new_password) < 8:
        flash("Password should be at least 8 characters.", "danger")
        return render_template("set_new_password.html", email=email)
    if new_password != confirm_password:
        flash("New password and confirm password must match.", "danger")
        return render_template("set_new_password.html", email=email)

    user = get_user_by_email(email)
    if not user:
        flash("User account not found for this email.", "danger")
        return render_template("set_new_password.html", email=email)

    new_password_hash = generate_password_hash(new_password)
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (new_password_hash, user["id"]),
            )
            cursor.execute(
                """
                UPDATE password_reset_requests
                SET status = 'Completed',
                    completed_at = CURRENT_TIMESTAMP,
                    admin_action_date = COALESCE(admin_action_date, CURRENT_TIMESTAMP)
                WHERE id = %s
                """,
                (latest_request["id"],),
            )
    finally:
        conn.close()

    flash(
        "Your password has been reset successfully. You may now log in with your new password.",
        "success",
    )
    return redirect(url_for("login"))


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    admin_redirect = _redirect_admin_to_dashboard()
    if admin_redirect:
        return admin_redirect
    ensure_database_tables()
    user = get_current_user()
    if not user:
        session.clear()
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("profile.html", user=user)

    full_name = request.form.get("full_name", "").strip()
    contact_number = request.form.get("contact_number", "").strip()
    address = request.form.get("address", "").strip()
    postcode = request.form.get("postcode", "").strip()
    state = request.form.get("state", "").strip()
    marital_status = request.form.get("marital_status", "").strip()
    family_count_raw = request.form.get("family_count", "").strip()
    profile_theme = request.form.get("profile_theme", "neon-purple").strip()
    avatar_size_raw = request.form.get("avatar_size", "120").strip()
    uploaded_profile_image = None

    if marital_status and marital_status not in MARITAL_STATUS_OPTIONS:
        flash("Invalid marital status.", "danger")
        return render_template("profile.html", user=user, form_data=request.form)
    if profile_theme not in PROFILE_THEME_OPTIONS:
        profile_theme = "neon-purple"

    if postcode and not re.fullmatch(r"\d{5}", postcode):
        flash("Postcode must be 5 digits.", "danger")
        return render_template("profile.html", user=user, form_data=request.form)
    if state and state not in get_state_choices():
        flash("Invalid state.", "danger")
        return render_template("profile.html", user=user, form_data=request.form)
    try:
        uploaded_profile_image = _save_profile_image(request.files.get("profile_image"))
    except ValueError as exc:
        error_message = _safe_str(str(exc), "Upload failed.")
        return render_template(
            "profile.html",
            user=user,
            form_data=request.form,
            popup_message=f"{error_message} Please try again upload.",
        )

    try:
        avatar_size = int(avatar_size_raw)
    except ValueError:
        flash("Avatar size must be a number.", "danger")
        return render_template("profile.html", user=user, form_data=request.form)
    avatar_size = max(80, min(200, avatar_size))

    family_count = None
    if family_count_raw:
        try:
            family_count = max(1, int(family_count_raw))
        except ValueError:
            flash("Family count must be a number.", "danger")
            return render_template("profile.html", user=user, form_data=request.form)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            if uploaded_profile_image:
                packet_limit = _fetch_db_max_allowed_packet_bytes(cursor)
                blob_size = len(uploaded_profile_image["blob"])
                safe_blob_limit = 0
                if packet_limit > 0:
                    safe_blob_limit = max(0, packet_limit - PROFILE_IMAGE_PACKET_MARGIN_BYTES)
                if safe_blob_limit > 0 and blob_size > safe_blob_limit:
                    packet_kb = max(1, packet_limit // 1024)
                    recommended_kb = max(1, safe_blob_limit // 1024)
                    popup_message = (
                        "Upload failed: image size exceeds the database maximum limit "
                        f"(max_allowed_packet ~ {packet_kb} KB). "
                        f"Please try again with a smaller image (recommended below {recommended_kb} KB)."
                    )
                    return render_template(
                        "profile.html",
                        user=user,
                        form_data=request.form,
                        popup_message=popup_message,
                    )
                cursor.execute(
                    """
                    UPDATE users
                    SET full_name = %s,
                        contact_number = %s,
                        address = %s,
                        postcode = %s,
                        state = %s,
                        marital_status = %s,
                        family_count = %s,
                        profile_image_url = NULL,
                        profile_image_blob = %s,
                        profile_image_mime = %s,
                        profile_image_name = %s,
                        profile_theme = %s,
                        avatar_size = %s
                    WHERE id = %s
                    """,
                    (
                        full_name or None,
                        contact_number or None,
                        address or None,
                        postcode or None,
                        state or None,
                        marital_status or None,
                        family_count,
                        uploaded_profile_image["blob"],
                        uploaded_profile_image["mime"],
                        uploaded_profile_image["name"],
                        profile_theme,
                        avatar_size,
                        user["id"],
                    ),
                )
            else:
                cursor.execute(
                    """
                    UPDATE users
                    SET full_name = %s,
                        contact_number = %s,
                        address = %s,
                        postcode = %s,
                        state = %s,
                        marital_status = %s,
                        family_count = %s,
                        profile_theme = %s,
                        avatar_size = %s
                    WHERE id = %s
                    """,
                    (
                        full_name or None,
                        contact_number or None,
                        address or None,
                        postcode or None,
                        state or None,
                        marital_status or None,
                        family_count,
                        profile_theme,
                        avatar_size,
                        user["id"],
                    ),
                )
    except pymysql.err.OperationalError as exc:
        error_code = _safe_int(exc.args[0] if exc.args else 0, default=0)
        if error_code == 1153:
            return render_template(
                "profile.html",
                user=user,
                form_data=request.form,
                popup_message=(
                    "Upload failed: image size exceeded the database maximum packet size. "
                    "Please try again with a smaller image."
                ),
            )
        else:
            return render_template(
                "profile.html",
                user=user,
                form_data=request.form,
                popup_message=(
                    "Unable to save profile image right now. "
                    "Please try again upload."
                ),
            )
    finally:
        conn.close()

    flash("Profile updated.", "success")
    return redirect(url_for("profile"))


@app.route("/profile-image/user/<int:user_id>", methods=["GET"])
def profile_image_by_user(user_id):
    uid = _safe_int(user_id, default=0)
    if uid <= 0:
        abort(404)
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT profile_image_blob, profile_image_mime
                FROM users
                WHERE id = %s
                LIMIT 1
                """,
                (uid,),
            )
            row = cursor.fetchone()
    finally:
        conn.close()
    if not row:
        abort(404)
    blob = row.get("profile_image_blob")
    if not blob:
        abort(404)
    mime = _safe_str(row.get("profile_image_mime"), "application/octet-stream") or "application/octet-stream"
    return Response(blob, mimetype=mime)


@app.route("/default-profile-image", methods=["GET"])
def default_profile_image():
    image_path = _resolve_default_profile_image_path()
    if image_path and os.path.exists(image_path):
        return send_file(image_path)
    return send_from_directory(os.path.join(BASE_DIR, "static", "images"), "default-profile.svg")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    admin_redirect = _redirect_admin_to_dashboard()
    if admin_redirect:
        return admin_redirect
    ensure_database_tables()
    if request.method == "GET":
        return render_template("predict.html", **prediction_form_context())

    try:
        input_payload, feature_df = parse_prediction_form(request.form)
        model = load_model()
        prediction = float(model.predict(feature_df)[0])
        user_id = session.get("user_id")
        save_prediction(input_payload, prediction, user_id=user_id)
        if not user_id:
            flash("Log in to save this result to your history.", "info")
        recommendations = generate_recommendations(input_payload, prediction)
        return render_template(
            "predict.html",
            prediction=round(prediction, 2),
            recommendations=recommendations,
            **prediction_form_context(submitted_data=input_payload),
        )
    except Exception as exc:
        flash(str(exc), "danger")
        return render_template(
            "predict.html",
            **prediction_form_context(submitted_data=request.form.to_dict()),
        )


@app.route("/search", methods=["GET"])
def search():
    admin_redirect = _redirect_admin_to_dashboard()
    if admin_redirect:
        return admin_redirect
    ensure_database_tables()
    filters = {}
    per_page_options = [25, 50, 100, 200]
    try:
        filters["min_price"] = parse_optional_filter_float(request.args, "min_price")
        filters["max_price"] = parse_optional_filter_float(request.args, "max_price")
        filters["state"] = request.args.get("state", "").strip() or None
        filters["property_type"] = parse_optional_filter_int(request.args, "property_type")
        filters["bedroom"] = parse_optional_filter_int(request.args, "bedroom")
        requested_page = parse_positive_int(request.args, "page", 1)
        requested_per_page = parse_positive_int(request.args, "per_page", 50)
    except ValueError:
        flash("Please enter valid numeric filter values.", "danger")
        filters = {"min_price": None, "max_price": None, "state": None, "property_type": None, "bedroom": None}
        requested_page = 1
        requested_per_page = 50

    if requested_per_page not in per_page_options:
        requested_per_page = 50

    search_result = search_listings(filters, page=requested_page, per_page=requested_per_page)
    rows = search_result["rows"]
    pagination = {
        "page": search_result["page"],
        "per_page": search_result["per_page"],
        "total_pages": search_result["total_pages"],
        "total_count": search_result["total_count"],
        "start_index": 0,
        "end_index": 0,
        "prev_url": None,
        "next_url": None,
        "page_links": [],
    }

    if pagination["total_count"] > 0:
        pagination["start_index"] = ((pagination["page"] - 1) * pagination["per_page"]) + 1
        pagination["end_index"] = min(pagination["page"] * pagination["per_page"], pagination["total_count"])

    query_base = {}
    for key, value in filters.items():
        if value is not None:
            query_base[key] = value
    query_base["per_page"] = pagination["per_page"]

    if pagination["page"] > 1:
        pagination["prev_url"] = url_for("search", **dict(query_base, page=pagination["page"] - 1))
    if pagination["page"] < pagination["total_pages"]:
        pagination["next_url"] = url_for("search", **dict(query_base, page=pagination["page"] + 1))

    page_window = 5
    start_page = max(1, pagination["page"] - 2)
    end_page = min(pagination["total_pages"], start_page + page_window - 1)
    start_page = max(1, end_page - page_window + 1)

    for page_number in range(start_page, end_page + 1):
        pagination["page_links"].append(
            {
                "number": page_number,
                "url": url_for("search", **dict(query_base, page=page_number)),
                "is_current": page_number == pagination["page"],
            }
        )

    llm_max_matches = max(0, _safe_int(os.getenv("LLM_MAX_MATCHES_PER_SEARCH", "200"), 200))
    search_nonce = int(time.time())
    seen_signatures = set()
    llm_error_count = 0
    llm_error_details = []
    llm_target_rows = min(llm_max_matches, len(rows))
    for idx, row in enumerate(rows):
        location_context = get_location_market_context(row.get("negeri"), row.get("area"))
        use_llm = idx < llm_max_matches
        match = generate_ai_property_match(
            row,
            use_llm=use_llm,
            row_rank=idx,
            search_nonce=search_nonce,
            variation_seed=0,
            location_context=location_context,
            allow_fallback=(not use_llm),
        )

        if use_llm:
            signature = _ai_match_text_signature(match)
            retries = 0
            while signature and signature in seen_signatures and retries < 2 and match.get("source") == "llm":
                retries += 1
                match = generate_ai_property_match(
                    row,
                    use_llm=True,
                    row_rank=idx,
                    search_nonce=search_nonce,
                    variation_seed=retries,
                    location_context=location_context,
                    allow_fallback=False,
                )
                signature = _ai_match_text_signature(match)

            if match.get("source") == "llm_error":
                llm_error_count += 1
                detail = _safe_str(match.get("error_detail"), "")
                if detail and detail not in llm_error_details:
                    llm_error_details.append(detail)

            if signature:
                seen_signatures.add(signature)

        row["ai_match"] = match

    if llm_target_rows > 0 and llm_error_count > 0:
        flash(
            f"LLM could not generate {llm_error_count} recommendation(s). Please check API configuration or retry.",
            "warning",
        )
        if llm_error_details:
            short_error = llm_error_details[0][:220]
            flash(f"LLM error detail: {short_error}", "warning")

    user_profile = get_current_user() if session.get("user_id") else None
    ai_overview = build_search_ai_overview(rows, filters=filters, user_profile=user_profile)
    return render_template(
        "search.html",
        rows=rows,
        ai_overview=ai_overview,
        filters=filters,
        pagination=pagination,
        per_page_options=per_page_options,
        state_choices=get_state_choices(),
        property_type_options=PROPERTY_TYPE_OPTIONS,
    )


@app.route("/property/<int:property_id>", methods=["GET"])
@login_required
def property_detail(property_id):
    admin_redirect = _redirect_admin_to_dashboard()
    if admin_redirect:
        return admin_redirect
    ensure_database_tables()
    row = get_property_by_id(property_id)
    if not row:
        flash("Property not found.", "warning")
        return redirect(url_for("search"))

    image_url = resolve_remote_property_image_url(property_id)
    nearest = {
        "hospital": None,
        "primary_school": None,
        "secondary_school": None,
    }
    if row.get("latitude") is not None and row.get("longitude") is not None:
        property_lat = float(row["latitude"])
        property_lon = float(row["longitude"])
        nearest["hospital"] = find_nearest_place(property_lat, property_lon, load_hospital_dataset())
        nearest["primary_school"] = find_nearest_place(
            property_lat, property_lon, load_primary_school_dataset()
        )
        nearest["secondary_school"] = find_nearest_place(
            property_lat, property_lon, load_secondary_school_dataset()
        )

    contact_number = extract_contact_number(row.get("description"))

    return render_template(
        "property_detail.html",
        row=row,
        image_url=image_url,
        nearest=nearest,
        contact_number=contact_number,
        furnishing_options=FURNISHING_OPTIONS,
        tenure_options=TENURE_OPTIONS,
        property_type_options=PROPERTY_TYPE_OPTIONS,
        unit_type_options=UNIT_TYPE_OPTIONS,
    )


@app.route("/history", methods=["GET"])
@login_required
def history():
    admin_redirect = _redirect_admin_to_dashboard()
    if admin_redirect:
        return admin_redirect
    ensure_database_tables()
    user_id = session.get("user_id")
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, input_json, predicted_price, created_at
                FROM predictions
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
    finally:
        conn.close()

    for row in rows:
        parsed_input = json.loads(row["input_json"]) if row["input_json"] else {}
        row["input_json"] = parsed_input
        row["input_display"] = format_prediction_input_for_display(parsed_input)
    return render_template("history.html", rows=rows)


@app.route("/init-db", methods=["GET"])
def init_db():
    ensure_database_tables()
    flash("Database initialized. Prediction and property tables are ready.", "success")
    return redirect(url_for("home"))


@app.route("/import-property-data", methods=["GET"])
def import_property_data():
    ensure_database_tables()
    try:
        inserted = import_property_dataset_into_db(replace_existing=True)
        flash(f"Property dataset imported successfully ({inserted} rows).", "success")
    except Exception as exc:
        flash(str(exc), "danger")
    return redirect(url_for("search"))


@app.route("/property-image/<path:filename>", methods=["GET"])
def property_image(filename):
    image_dir = _safe_str(PROPERTY_IMAGE_DIR, "")
    if not image_dir or not os.path.isdir(image_dir):
        abort(404)

    safe_name = os.path.basename(filename)
    if safe_name != filename:
        abort(404)

    allowed_extensions = (".jpg", ".jpeg", ".png", ".webp")
    if not safe_name.lower().endswith(allowed_extensions):
        abort(404)

    file_path = os.path.join(image_dir, safe_name)
    if not os.path.exists(file_path):
        abort(404)

    return send_from_directory(image_dir, safe_name)


MOCK_ADMIN_CUSTOMERS = [
    {
        "customer_id": 101,
        "name": "Aisyah Rahman",
        "email": "aisyah.rahman@example.com",
        "phone": "0123456789",
        "marital_status": "Married",
        "family_count": 4,
        "registered_date": "15 Jun 2026",
        "status": "Active",
    },
    {
        "customer_id": 102,
        "name": "Daniel Tan",
        "email": "daniel.tan@example.com",
        "phone": "",
        "marital_status": "Single",
        "family_count": 1,
        "registered_date": "20 Jun 2026",
        "status": "Incomplete Profile",
    },
]
MOCK_ADMIN_PREDICTIONS = [
    {
        "prediction_id": 501,
        "customer_name": "Aisyah Rahman",
        "customer_email": "aisyah.rahman@example.com",
        "property_type": "Terrace House",
        "property_type_code": 1,
        "location": "Shah Alam, Selangor",
        "state": "Selangor",
        "area": "Shah Alam",
        "built_up_size": 1650,
        "land_size": 1400,
        "bedrooms": 4,
        "bathrooms": 3,
        "tenure": "Freehold",
        "tenure_code": 2,
        "estimated_value": 628000,
        "prediction_date": "06 Jul 2026, 10:15 AM",
        "created_at_raw": None,
        "payload": {
            "Property_Type": 1,
            "Built_Up_SF": 1650,
            "Land_Size": 1400,
            "Bedroom": 4,
            "Bathroom": 3,
            "Tenure": 2,
            "state": "Selangor",
            "area_text": "Shah Alam",
        },
    },
    {
        "prediction_id": 502,
        "customer_name": "Daniel Tan",
        "customer_email": "daniel.tan@example.com",
        "property_type": "Condominium / Apartment / Serviced Residence",
        "property_type_code": 3,
        "location": "George Town, Penang",
        "state": "Penang",
        "area": "George Town",
        "built_up_size": 980,
        "land_size": 980,
        "bedrooms": 3,
        "bathrooms": 2,
        "tenure": "Leasehold",
        "tenure_code": 1,
        "estimated_value": 512400,
        "prediction_date": "05 Jul 2026, 03:42 PM",
        "created_at_raw": None,
        "payload": {
            "Property_Type": 3,
            "Built_Up_SF": 980,
            "Land_Size": 980,
            "Bedroom": 3,
            "Bathroom": 2,
            "Tenure": 1,
            "state": "Penang",
            "area_text": "George Town",
        },
    },
]


def _admin_safe_next_url(next_url):
    if not next_url or not next_url.startswith("/") or next_url.startswith("//"):
        return url_for("admin_dashboard")
    return next_url


def _build_admin_pagination(rows, page=1, per_page=25):
    total_rows = len(rows)
    safe_per_page = max(1, _safe_int(per_page, 25))
    total_pages = max(1, (total_rows + safe_per_page - 1) // safe_per_page) if total_rows else 1
    safe_page = max(1, _safe_int(page, 1))
    if safe_page > total_pages:
        safe_page = total_pages

    start_index = (safe_page - 1) * safe_per_page
    end_index = start_index + safe_per_page
    paged_rows = rows[start_index:end_index]

    page_window = 2
    page_start = max(1, safe_page - page_window)
    page_end = min(total_pages, safe_page + page_window)
    page_numbers = list(range(page_start, page_end + 1))
    pagination = {
        "page": safe_page,
        "per_page": safe_per_page,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "start_row": start_index + 1 if total_rows else 0,
        "end_row": min(end_index, total_rows),
        "has_prev": safe_page > 1,
        "has_next": safe_page < total_pages,
        "prev_page": safe_page - 1,
        "next_page": safe_page + 1,
        "page_numbers": page_numbers,
        "show_first_page": 1 not in page_numbers,
        "show_last_page": total_pages not in page_numbers,
    }
    return paged_rows, pagination


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Please log in as admin to continue.", "warning")
            return redirect(url_for("login", next=request.path, login_as="admin"))
        return view(*args, **kwargs)

    return wrapped


def _admin_datetime_label(raw_value, include_time=True):
    if hasattr(raw_value, "strftime"):
        if include_time:
            return raw_value.strftime("%d %b %Y, %I:%M %p")
        return raw_value.strftime("%d %b %Y")
    text = _safe_str(raw_value, "")
    if not text:
        return "-"
    return text


def _admin_decode_prediction_payload(raw_value):
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if raw_value is None:
        return {}
    try:
        return json.loads(raw_value)
    except Exception:
        return {}


def _admin_prediction_display_fields(payload):
    data = payload or {}
    property_type_code = _safe_int(data.get("Property_Type"), default=0)
    tenure_code = _safe_int(data.get("Tenure"), default=0)
    area = _safe_str(data.get("area_text"), "")
    state = _safe_str(data.get("state") or data.get("negeri"), "")
    if area and state:
        location = f"{area}, {state}"
    else:
        location = area or state or "-"
    return {
        "property_type": PROPERTY_TYPE_OPTIONS.get(property_type_code, "Unknown"),
        "property_type_code": property_type_code,
        "location": location,
        "state": state or "-",
        "area": area or "-",
        "built_up_size": _safe_float(data.get("Built_Up_SF"), default=0.0),
        "land_size": _safe_float(data.get("Land_Size"), default=0.0),
        "bedrooms": _safe_int(data.get("Bedroom"), default=0),
        "bathrooms": _safe_int(data.get("Bathroom"), default=0),
        "tenure": TENURE_OPTIONS.get(tenure_code, "Unknown"),
        "tenure_code": tenure_code,
    }


def _admin_filter_customer_rows(rows, search_text="", status_filter="all"):
    search = _safe_str(search_text, "").strip().lower()
    status = _safe_str(status_filter, "all").strip().lower()
    filtered = []
    for row in rows:
        row_status = _safe_str(row.get("status"), "Active")
        if status != "all":
            if status == "active" and row_status.lower() != "active":
                continue
            if status == "incomplete" and row_status.lower() != "incomplete profile":
                continue
        if search:
            haystack = " ".join(
                [
                    str(row.get("customer_id", "")),
                    _safe_str(row.get("name"), ""),
                    _safe_str(row.get("email"), ""),
                    _safe_str(row.get("phone"), ""),
                ]
            ).lower()
            if search not in haystack:
                continue
        filtered.append(row)
    return filtered


def _admin_filter_prediction_rows(rows, search_text="", state_filter="all"):
    search = _safe_str(search_text, "").strip().lower()
    state = _safe_str(state_filter, "all").strip().lower()
    filtered = []
    for row in rows:
        if state != "all":
            row_state = _safe_str(row.get("state"), "").strip().lower()
            if row_state != state:
                continue
        if search:
            haystack = " ".join(
                [
                    str(row.get("prediction_id", "")),
                    _safe_str(row.get("customer_name"), ""),
                    _safe_str(row.get("customer_email"), ""),
                    _safe_str(row.get("property_type"), ""),
                    _safe_str(row.get("location"), ""),
                ]
            ).lower()
            if search not in haystack:
                continue
        filtered.append(row)
    return filtered


def _admin_filter_property_rows(rows, search_text="", state_filter="all", property_type_filter="all"):
    search = _safe_str(search_text, "").strip().lower()
    state = _safe_str(state_filter, "all").strip().lower()
    property_type = _safe_str(property_type_filter, "all").strip()
    filtered = []
    for row in rows:
        if state != "all":
            if _safe_str(row.get("state"), "").strip().lower() != state:
                continue
        if property_type != "all":
            if str(_safe_int(row.get("property_type_code"), default=0)) != property_type:
                continue
        if search:
            haystack = " ".join(
                [
                    str(row.get("property_id", "")),
                    _safe_str(row.get("title"), ""),
                    _safe_str(row.get("location"), ""),
                    _safe_str(row.get("state"), ""),
                ]
            ).lower()
            if search not in haystack:
                continue
        filtered.append(row)
    return filtered


def _fetch_admin_customers(search_text="", status_filter="all"):
    rows = []
    use_mock = False
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, full_name, email, contact_number, marital_status, family_count, created_at
                FROM users
                ORDER BY created_at DESC
                LIMIT 2000
                """
            )
            db_rows = cursor.fetchall()
        for row in db_rows:
            customer_id = _safe_int(row.get("id"), default=0)
            full_name = _safe_str(row.get("full_name"), "").strip() or f"Customer {customer_id}"
            phone = _safe_str(row.get("contact_number"), "").strip()
            marital_key = _safe_str(row.get("marital_status"), "").strip().lower()
            marital_status = MARITAL_STATUS_OPTIONS.get(marital_key, "-")
            family_count = _safe_int(row.get("family_count"), default=0)
            rows.append(
                {
                    "customer_id": customer_id,
                    "name": full_name,
                    "email": _safe_str(row.get("email"), "-"),
                    "phone": phone,
                    "marital_status": marital_status,
                    "family_count": family_count if family_count > 0 else "-",
                    "registered_date": _admin_datetime_label(row.get("created_at"), include_time=False),
                    "status": "Active" if phone else "Incomplete Profile",
                }
            )
    except Exception:
        rows = []
        for item in MOCK_ADMIN_CUSTOMERS:
            row = dict(item)
            row["marital_status"] = _safe_str(row.get("marital_status"), "-") or "-"
            family_count = _safe_int(row.get("family_count"), default=0)
            row["family_count"] = family_count if family_count > 0 else "-"
            rows.append(row)
        use_mock = True
    finally:
        if conn:
            conn.close()
    return _admin_filter_customer_rows(rows, search_text=search_text, status_filter=status_filter), use_mock


def _fetch_admin_predictions(search_text="", state_filter="all"):
    rows = []
    use_mock = False
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.input_json,
                    p.predicted_price,
                    p.created_at,
                    p.user_id,
                    u.full_name,
                    u.email
                FROM predictions p
                LEFT JOIN users u ON u.id = p.user_id
                ORDER BY p.created_at DESC
                LIMIT 3000
                """
            )
            db_rows = cursor.fetchall()
        for row in db_rows:
            payload = _admin_decode_prediction_payload(row.get("input_json"))
            meta = _admin_prediction_display_fields(payload)
            customer_name = _safe_str(row.get("full_name"), "").strip() or "Guest User"
            rows.append(
                {
                    "prediction_id": _safe_int(row.get("id"), default=0),
                    "customer_name": customer_name,
                    "customer_email": _safe_str(row.get("email"), "-"),
                    "property_type": meta["property_type"],
                    "property_type_code": meta["property_type_code"],
                    "location": meta["location"],
                    "state": meta["state"],
                    "area": meta["area"],
                    "built_up_size": meta["built_up_size"],
                    "land_size": meta["land_size"],
                    "bedrooms": meta["bedrooms"],
                    "bathrooms": meta["bathrooms"],
                    "tenure": meta["tenure"],
                    "tenure_code": meta["tenure_code"],
                    "estimated_value": _safe_float(row.get("predicted_price"), default=0.0),
                    "prediction_date": _admin_datetime_label(row.get("created_at"), include_time=True),
                    "created_at_raw": row.get("created_at"),
                    "payload": payload,
                    "user_id": _safe_int(row.get("user_id"), default=0),
                }
            )
    except Exception:
        rows = [dict(item) for item in MOCK_ADMIN_PREDICTIONS]
        use_mock = True
    finally:
        if conn:
            conn.close()
    return _admin_filter_prediction_rows(rows, search_text=search_text, state_filter=state_filter), use_mock


def _fetch_admin_property_rows(search_text="", state_filter="all", property_type_filter="all"):
    rows = []
    use_mock = False
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id, title, area, negeri, property_type,
                    built_up_sf, land_size, bedroom, bathroom,
                    car_park, tenure, listing_price, image_url, created_at
                FROM property_listings
                ORDER BY id DESC
                LIMIT 5000
                """
            )
            db_rows = cursor.fetchall()
        for row in db_rows:
            property_id = _safe_int(row.get("id"), default=0)
            image_url = _safe_str(row.get("image_url"), "").strip() or None
            rows.append(
                {
                    "property_id": property_id,
                    "title": _safe_str(row.get("title"), ""),
                    "location": _safe_str(row.get("area"), "-"),
                    "state": _safe_str(row.get("negeri"), "-"),
                    "property_type": PROPERTY_TYPE_OPTIONS.get(
                        _safe_int(row.get("property_type"), default=0), "Unknown"
                    ),
                    "property_type_code": _safe_int(row.get("property_type"), default=0),
                    "built_up_size": _safe_float(row.get("built_up_sf"), default=0.0),
                    "land_size": _safe_float(row.get("land_size"), default=0.0),
                    "bedrooms": _safe_int(row.get("bedroom"), default=0),
                    "bathrooms": _safe_int(row.get("bathroom"), default=0),
                    "car_park": _safe_int(row.get("car_park"), default=0),
                    "tenure": TENURE_OPTIONS.get(_safe_int(row.get("tenure"), default=0), "Unknown"),
                    "actual_price": _safe_float(row.get("listing_price"), default=0.0),
                    "created_date": _admin_datetime_label(row.get("created_at"), include_time=False),
                    "image_url": image_url,
                }
            )
    except Exception:
        use_mock = True
        rows = []
        for listing in SEED_PROPERTY_LISTINGS:
            rows.append(
                {
                    "property_id": 0,
                    "title": _safe_str(listing.get("title"), ""),
                    "location": _safe_str(listing.get("area"), "-"),
                    "state": _safe_str(listing.get("negeri"), "-"),
                    "property_type": PROPERTY_TYPE_OPTIONS.get(
                        _safe_int(listing.get("property_type"), default=0), "Unknown"
                    ),
                    "property_type_code": _safe_int(listing.get("property_type"), default=0),
                    "built_up_size": _safe_float(listing.get("built_up_sf"), default=0.0),
                    "land_size": _safe_float(listing.get("land_size"), default=0.0),
                    "bedrooms": _safe_int(listing.get("bedroom"), default=0),
                    "bathrooms": _safe_int(listing.get("bathroom"), default=0),
                    "car_park": _safe_int(listing.get("car_park"), default=0),
                    "tenure": TENURE_OPTIONS.get(_safe_int(listing.get("tenure"), default=0), "Unknown"),
                    "actual_price": _safe_float(listing.get("listing_price"), default=0.0),
                    "created_date": "-",
                    "image_url": None,
                }
            )
    finally:
        if conn:
            conn.close()
    return _admin_filter_property_rows(
        rows,
        search_text=search_text,
        state_filter=state_filter,
        property_type_filter=property_type_filter,
    ), use_mock


def _get_admin_count_metrics():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM users")
            total_customers = _safe_int(cursor.fetchone().get("total"), default=0)
            cursor.execute("SELECT COUNT(*) AS total FROM property_listings")
            total_valuations = _safe_int(cursor.fetchone().get("total"), default=0)
            cursor.execute("SELECT COUNT(*) AS total FROM predictions")
            total_predictions = _safe_int(cursor.fetchone().get("total"), default=0)
        return {
            "total_customers": total_customers,
            "total_valuations": total_valuations,
            "total_predictions": total_predictions,
            "used_mock": False,
        }
    except Exception:
        return {
            "total_customers": len(MOCK_ADMIN_CUSTOMERS),
            "total_valuations": len(SEED_PROPERTY_LISTINGS),
            "total_predictions": len(MOCK_ADMIN_PREDICTIONS),
            "used_mock": True,
        }
    finally:
        if conn:
            conn.close()


def _build_prediction_trend(predictions):
    buckets = {}
    for row in predictions:
        raw = row.get("created_at_raw")
        label = None
        if hasattr(raw, "strftime"):
            label = raw.strftime("%b %Y")
        elif raw:
            label = _safe_str(raw, "")[:7]
        if not label:
            label = "Recent"
        buckets[label] = buckets.get(label, 0) + 1
    if not buckets:
        return {"labels": ["Recent"], "series": [0]}
    labels = list(buckets.keys())[-6:]
    series = [buckets[label] for label in labels]
    return {"labels": labels, "series": series}


def _resolve_last_training_date():
    env_date = _safe_str(os.getenv("MODEL_LAST_TRAINING_DATE"), "")
    model_path = _resolve_project_path(os.getenv("MODEL_PATH", "models/best_model.pkl"))
    if os.path.exists(model_path):
        return datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%d %b %Y")
    return env_date or "Not available"


def _format_model_param_value(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        return f"{value:.6g}"
    return str(value)


MODEL_PARAM_DESCRIPTIONS = {
    "objective": "Defines the prediction goal and the error metric minimized during model training.",
    "n_estimators": "Controls how many boosting trees are built; more trees can improve accuracy but increase training time.",
    "learning_rate": "Controls how much each tree updates the prediction; lower values improve stability but usually need more trees.",
    "max_depth": "Limits tree depth to control model complexity and reduce overfitting on property data.",
    "subsample": "Uses a random subset of training rows for each tree to improve generalization.",
    "colsample_bytree": "Uses a random subset of features for each tree to improve robustness across property attributes.",
    "random_state": "Fixes randomness so training and evaluation results remain consistent and reproducible.",
    "n_jobs": "Controls parallel processing during training; -1 uses all available CPU cores for faster training.",
}


def _extract_active_model_parameter_items():
    fallback_params = dict(XGBOOST_AFTER_TUNED_PARAMS)
    model_params = {}
    try:
        loaded_model = load_model()
        estimator = loaded_model
        if hasattr(loaded_model, "named_steps") and "model" in loaded_model.named_steps:
            estimator = loaded_model.named_steps["model"]
        elif hasattr(loaded_model, "steps") and loaded_model.steps:
            estimator = loaded_model.steps[-1][1]
        if hasattr(estimator, "get_params"):
            model_params = estimator.get_params() or {}
    except Exception:
        model_params = {}

    resolved_params = dict(fallback_params)
    for key in fallback_params:
        if key in model_params and model_params.get(key) is not None:
            resolved_params[key] = model_params[key]

    items = []
    for key in fallback_params:
        value = resolved_params.get(key)
        if value is None:
            continue
        items.append(
            {
                "name": key,
                "value": _format_model_param_value(value),
                "description": MODEL_PARAM_DESCRIPTIONS.get(key, ""),
            }
        )
    return items


def _get_admin_model_metrics():
    metrics = get_model_quality_metrics()
    model_name = _safe_str(os.getenv("MODEL_NAME"), "") or "XGBoost Regressor (After Tuning)"
    parameter_items = _extract_active_model_parameter_items()
    if metrics:
        return {
            "model_name": model_name,
            "r2": _safe_float(metrics.get("r2"), default=0.0),
            "rmse": _safe_float(metrics.get("rmse"), default=0.0),
            "mae": _safe_float(metrics.get("mae"), default=0.0),
            "reliability": _safe_str(metrics.get("reliability"), "Moderate reliability"),
            "evaluation_note": _safe_str(metrics.get("evaluation_note"), ""),
            "sample_size": _safe_int(metrics.get("sample_size"), default=0),
            "last_training_date": _resolve_last_training_date(),
            "parameter_items": parameter_items,
            "used_mock": False,
        }
    return {
        "model_name": model_name,
        "r2": 0.8721,
        "rmse": 58912.43,
        "mae": 42108.76,
        "reliability": "Moderate reliability",
        "evaluation_note": "Mock metrics shown because model evaluation data is unavailable.",
        "sample_size": 0,
        "last_training_date": _resolve_last_training_date(),
        "parameter_items": parameter_items,
        "used_mock": True,
    }


def _parse_admin_property_form(form_data):
    title = _safe_str(form_data.get("title"), "").strip()
    area = _safe_str(form_data.get("area"), "").strip()
    negeri = _safe_str(form_data.get("negeri"), "").strip()
    if not title:
        raise ValueError("Property title is required.")
    if not area:
        raise ValueError("Location/area is required.")
    if not negeri:
        raise ValueError("State is required.")

    property_type = parse_int_choice(form_data, "property_type", PROPERTY_TYPE_OPTIONS)
    furnishing = parse_int_choice(form_data, "furnishing", FURNISHING_OPTIONS)
    tenure = parse_int_choice(form_data, "tenure", TENURE_OPTIONS)
    unit_type = parse_int_choice(form_data, "unit_type", UNIT_TYPE_OPTIONS)

    built_up_sf = parse_float(form_data, "built_up_sf", min_value=100.0, max_value=100000.0)
    land_size = parse_float(form_data, "land_size", min_value=100.0, max_value=200000.0)
    listing_price = parse_float(form_data, "listing_price", min_value=10000.0, max_value=100000000.0)

    bedroom = int(parse_float(form_data, "bedroom", min_value=0.0, max_value=30.0))
    bathroom = int(parse_float(form_data, "bathroom", min_value=0.0, max_value=30.0))
    car_park = int(parse_float(form_data, "car_park", min_value=0.0, max_value=20.0))

    lat_raw = _safe_str(form_data.get("latitude"), "").strip()
    lon_raw = _safe_str(form_data.get("longitude"), "").strip()
    latitude = float(lat_raw) if lat_raw else None
    longitude = float(lon_raw) if lon_raw else None
    image_url = _safe_str(form_data.get("image_url"), "").strip()
    if image_url and not image_url.lower().startswith(("http://", "https://")):
        raise ValueError("Image URL must start with http:// or https://")

    return {
        "title": title,
        "area": area,
        "negeri": negeri,
        "property_type": property_type,
        "built_up_sf": built_up_sf,
        "land_size": land_size,
        "bedroom": bedroom,
        "bathroom": bathroom,
        "car_park": car_park,
        "furnishing": furnishing,
        "tenure": tenure,
        "unit_type": unit_type,
        "listing_price": listing_price,
        "latitude": latitude,
        "longitude": longitude,
        "image_url": image_url or None,
    }


def _find_prediction_record_by_id(prediction_id):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.input_json,
                    p.predicted_price,
                    p.created_at,
                    p.user_id,
                    u.full_name,
                    u.email
                FROM predictions p
                LEFT JOIN users u ON u.id = p.user_id
                WHERE p.id = %s
                LIMIT 1
                """,
                (prediction_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        payload = _admin_decode_prediction_payload(row.get("input_json"))
        meta = _admin_prediction_display_fields(payload)
        return {
            "prediction_id": _safe_int(row.get("id"), default=0),
            "customer_name": _safe_str(row.get("full_name"), "").strip() or "Guest User",
            "customer_email": _safe_str(row.get("email"), "-"),
            "estimated_value": _safe_float(row.get("predicted_price"), default=0.0),
            "prediction_date": _admin_datetime_label(row.get("created_at"), include_time=True),
            "created_at_raw": row.get("created_at"),
            "payload": payload,
            "meta": meta,
        }
    except Exception:
        for item in MOCK_ADMIN_PREDICTIONS:
            if _safe_int(item.get("prediction_id"), default=0) == prediction_id:
                meta = _admin_prediction_display_fields(item.get("payload"))
                clone = dict(item)
                clone["meta"] = meta
                return clone
        return None
    finally:
        if conn:
            conn.close()


def _find_reference_listing_for_prediction(meta):
    area = _safe_str(meta.get("area"), "").strip()
    state = _safe_str(meta.get("state"), "").strip()
    property_type_code = _safe_int(meta.get("property_type_code"), default=0)
    built_up_size = _safe_float(meta.get("built_up_size"), default=0.0)
    conn = None
    try:
        conn = get_db_connection()
        conditions = []
        params = []
        if area and area != "-":
            conditions.append("area = %s")
            params.append(area)
        if state and state != "-":
            conditions.append("negeri = %s")
            params.append(state)
        if property_type_code > 0:
            conditions.append("property_type = %s")
            params.append(property_type_code)
        where_sql = " AND ".join(conditions) if conditions else "1=1"
        order_sql = "ORDER BY id DESC"
        if built_up_size > 0:
            order_sql = "ORDER BY ABS(built_up_sf - %s) ASC, id DESC"
            params.append(built_up_size)
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM property_listings
                WHERE {where_sql}
                {order_sql}
                LIMIT 1
                """,
                params,
            )
            row = cursor.fetchone()
        return normalize_property_row(row) if row else None
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


def _find_nearest_school(property_lat, property_lon):
    primary = find_nearest_place(property_lat, property_lon, load_primary_school_dataset())
    secondary = find_nearest_place(property_lat, property_lon, load_secondary_school_dataset())
    candidates = []
    if primary:
        candidate = dict(primary)
        candidate["category"] = "Primary School"
        candidates.append(candidate)
    if secondary:
        candidate = dict(secondary)
        candidate["category"] = "Secondary School"
        candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda item: _safe_float(item.get("distance_km"), default=99999.0))
    return candidates[0]


def _build_prediction_facilities(reference_property):
    facilities = {
        "hospital": None,
        "school": None,
        "mall": None,
        "public_transport": None,
    }
    if not reference_property:
        return facilities
    lat = reference_property.get("latitude")
    lon = reference_property.get("longitude")
    if lat is None or lon is None:
        return facilities
    try:
        property_lat = float(lat)
        property_lon = float(lon)
    except Exception:
        return facilities
    facilities["hospital"] = find_nearest_place(property_lat, property_lon, load_hospital_dataset())
    facilities["school"] = _find_nearest_school(property_lat, property_lon)
    return facilities


@app.route("/admin", methods=["GET"])
def admin_index():
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("login", next=url_for("admin_dashboard"), login_as="admin"))


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    next_url = _admin_safe_next_url(request.args.get("next") or request.form.get("next"))
    return redirect(url_for("login", next=next_url, login_as="admin"))


@app.route("/admin/logout", methods=["GET"])
def admin_logout():
    session.clear()
    flash("Admin session ended.", "info")
    return redirect(url_for("login", login_as="admin"))


@app.route("/admin/dashboard", methods=["GET"])
@admin_required
def admin_dashboard():
    ensure_database_tables()
    counts = _get_admin_count_metrics()
    predictions, prediction_mock = _fetch_admin_predictions()
    recent_activity = predictions[:8]
    reset_requests = fetch_password_reset_requests()
    recent_reset_requests = reset_requests[:8]
    pending_reset_requests = count_pending_password_reset_requests()
    trend = _build_prediction_trend(predictions)
    model_metrics = _get_admin_model_metrics()
    summary_chart = {
        "labels": [
            "Customers",
            "Property Valuations",
            "Prediction Records",
            "Pending Password Reset Requests",
        ],
        "series": [
            counts["total_customers"],
            counts["total_valuations"],
            counts["total_predictions"],
            pending_reset_requests,
        ],
    }
    return render_template(
        "admin/dashboard.html",
        page_title="Admin Dashboard",
        active_admin_page="dashboard",
        total_customers=counts["total_customers"],
        total_valuations=counts["total_valuations"],
        total_predictions=counts["total_predictions"],
        pending_reset_requests=pending_reset_requests,
        model_metrics=model_metrics,
        recent_activity=recent_activity,
        recent_reset_requests=recent_reset_requests,
        summary_chart=summary_chart,
        trend_chart=trend,
        using_mock_data=counts["used_mock"] or prediction_mock or model_metrics.get("used_mock"),
    )


@app.route("/admin/customers", methods=["GET"])
@admin_required
def admin_customers():
    ensure_database_tables()
    search_text = _safe_str(request.args.get("q"), "").strip()
    status_filter = _safe_str(request.args.get("status"), "all").strip().lower()
    per_page_options = [25, 50, 75]
    try:
        page = parse_positive_int(request.args, "page", 1)
    except Exception:
        page = 1
    try:
        per_page = parse_positive_int(request.args, "per_page", per_page_options[0])
    except Exception:
        per_page = per_page_options[0]
    if per_page not in per_page_options:
        per_page = per_page_options[0]
    rows, use_mock = _fetch_admin_customers(search_text=search_text, status_filter=status_filter)
    paged_rows, pagination = _build_admin_pagination(rows, page=page, per_page=per_page)
    return render_template(
        "admin/customers.html",
        page_title="Manage Customers",
        active_admin_page="customers",
        rows=paged_rows,
        filters={"q": search_text, "status": status_filter or "all"},
        per_page_options=per_page_options,
        pagination=pagination,
        using_mock_data=use_mock,
    )


@app.route("/admin/customers/<int:customer_id>", methods=["GET"])
@admin_required
def admin_customer_view(customer_id):
    ensure_database_tables()
    conn = None
    customer = None
    predictions_total = 0
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, full_name, email, contact_number, address, postcode, state,
                       marital_status, family_count, profile_image_url,
                       (profile_image_blob IS NOT NULL) AS has_profile_image_blob, created_at
                FROM users
                WHERE id = %s
                LIMIT 1
                """,
                (customer_id,),
            )
            row = cursor.fetchone()
            if row:
                cursor.execute("SELECT COUNT(*) AS total FROM predictions WHERE user_id = %s", (customer_id,))
                predictions_total = _safe_int(cursor.fetchone().get("total"), default=0)
                phone = _safe_str(row.get("contact_number"), "").strip()
                full_name = _safe_str(row.get("full_name"), "").strip() or f"Customer {customer_id}"
                name_parts = [part for part in full_name.split(" ") if part]
                initials = "".join(part[0].upper() for part in name_parts[:2]) or "CU"
                has_blob = bool(_safe_int(row.get("has_profile_image_blob"), default=0))
                profile_image_src = (
                    url_for("profile_image_by_user", user_id=_safe_int(row.get("id"), default=0))
                    if has_blob
                    else None
                )
                marital_key = _safe_str(row.get("marital_status"), "").strip().lower()
                marital_status = MARITAL_STATUS_OPTIONS.get(marital_key, "-")
                family_count = _safe_int(row.get("family_count"), default=0)
                customer = {
                    "customer_id": _safe_int(row.get("id"), default=0),
                    "name": full_name,
                    "initials": initials,
                    "email": _safe_str(row.get("email"), "-"),
                    "phone": phone or "-",
                    "marital_status": marital_status,
                    "family_count": family_count if family_count > 0 else "-",
                    "address": _safe_str(row.get("address"), "-") or "-",
                    "postcode": _safe_str(row.get("postcode"), "-") or "-",
                    "state": _safe_str(row.get("state"), "-") or "-",
                    "profile_image_src": profile_image_src,
                    "registered_date": _admin_datetime_label(row.get("created_at"), include_time=False),
                    "status": "Active" if phone else "Incomplete Profile",
                }
    except Exception:
        customer = None
    finally:
        if conn:
            conn.close()

    if not customer:
        for mock in MOCK_ADMIN_CUSTOMERS:
            if _safe_int(mock.get("customer_id"), default=0) == customer_id:
                customer = dict(mock)
                name_parts = [part for part in _safe_str(customer.get("name"), "").split(" ") if part]
                customer["initials"] = "".join(part[0].upper() for part in name_parts[:2]) or "CU"
                customer["profile_image_src"] = None
                customer["address"] = "-"
                customer["postcode"] = "-"
                customer["state"] = "-"
                customer["marital_status"] = _safe_str(customer.get("marital_status"), "-") or "-"
                family_count = _safe_int(customer.get("family_count"), default=0)
                customer["family_count"] = family_count if family_count > 0 else "-"
                predictions_total = 1
                break

    if not customer:
        flash("Customer not found.", "warning")
        return redirect(url_for("admin_customers"))
    return render_template(
        "admin/customer_view.html",
        page_title="Customer Details",
        active_admin_page="customers",
        customer=customer,
        predictions_total=predictions_total,
    )


@app.route("/admin/customers/<int:customer_id>/delete", methods=["POST"])
@admin_required
def admin_customer_delete(customer_id):
    ensure_database_tables()
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("UPDATE predictions SET user_id = NULL WHERE user_id = %s", (customer_id,))
            cursor.execute("DELETE FROM users WHERE id = %s", (customer_id,))
        flash("Customer deleted successfully.", "success")
    except Exception as exc:
        flash(f"Unable to delete customer: {exc}", "danger")
    finally:
        if conn:
            conn.close()
    return redirect(url_for("admin_customers"))


@app.route("/admin/predictions", methods=["GET"])
@admin_required
def admin_predictions():
    ensure_database_tables()
    search_text = _safe_str(request.args.get("q"), "").strip()
    state_filter = _safe_str(request.args.get("state"), "all").strip()
    per_page_options = [25, 50, 75]
    try:
        page = parse_positive_int(request.args, "page", 1)
    except Exception:
        page = 1
    try:
        per_page = parse_positive_int(request.args, "per_page", per_page_options[0])
    except Exception:
        per_page = per_page_options[0]
    if per_page not in per_page_options:
        per_page = per_page_options[0]
    rows, use_mock = _fetch_admin_predictions(search_text=search_text, state_filter=state_filter)
    paged_rows, pagination = _build_admin_pagination(rows, page=page, per_page=per_page)
    return render_template(
        "admin/predictions.html",
        page_title="Prediction Records",
        active_admin_page="predictions",
        rows=paged_rows,
        filters={"q": search_text, "state": state_filter or "all"},
        state_choices=get_state_choices(),
        per_page_options=per_page_options,
        pagination=pagination,
        using_mock_data=use_mock,
    )


@app.route("/admin/predictions/<int:prediction_id>", methods=["GET"])
@admin_required
def admin_prediction_detail(prediction_id):
    ensure_database_tables()
    prediction = _find_prediction_record_by_id(prediction_id)
    if not prediction:
        flash("Prediction record not found.", "warning")
        return redirect(url_for("admin_predictions"))
    meta = prediction.get("meta") or _admin_prediction_display_fields(prediction.get("payload"))
    reference_property = _find_reference_listing_for_prediction(meta)
    facilities = _build_prediction_facilities(reference_property)
    return render_template(
        "admin/prediction_detail.html",
        page_title="Prediction Details",
        active_admin_page="predictions",
        prediction=prediction,
        meta=meta,
        reference_property=reference_property,
        facilities=facilities,
    )


@app.route("/admin/password-reset-requests", methods=["GET"])
@admin_required
def admin_password_reset_requests():
    ensure_database_tables()
    search_text = _safe_str(request.args.get("q"), "").strip()
    status_filter = _safe_str(request.args.get("status"), "all").strip().title()
    if status_filter not in {"All", "Pending", "Approved", "Rejected", "Completed"}:
        status_filter = "All"
    rows = fetch_password_reset_requests(search_text=search_text, status_filter=status_filter)
    return render_template(
        "admin/password_reset_requests.html",
        page_title="Password Reset Requests",
        active_admin_page="password-reset-requests",
        rows=rows,
        status_options=["All", "Pending", "Approved", "Rejected", "Completed"],
        filters={"q": search_text, "status": status_filter},
    )


@app.route("/admin/password-reset-requests/<int:request_id>", methods=["GET"])
@admin_required
def admin_password_reset_request_detail(request_id):
    ensure_database_tables()
    row = get_password_reset_request_by_id(request_id)
    if not row:
        flash("Password reset request not found.", "warning")
        return redirect(url_for("admin_password_reset_requests"))
    return render_template(
        "admin/password_reset_request_detail.html",
        page_title="Password Reset Request Details",
        active_admin_page="password-reset-requests",
        row=row,
    )


@app.route("/admin/password-reset-requests/<int:request_id>/approve", methods=["POST"])
@admin_required
def admin_password_reset_approve(request_id):
    ensure_database_tables()
    approved = update_password_reset_request_status(
        request_id,
        "Approved",
        action_by=_safe_str(session.get("admin_email"), "").strip(),
    )
    if approved:
        flash("Password reset request has been approved.", "success")
    else:
        flash("Unable to approve this request. It may already be completed or unavailable.", "warning")
    return redirect(url_for("admin_password_reset_requests"))


@app.route("/admin/password-reset-requests/<int:request_id>/reject", methods=["POST"])
@admin_required
def admin_password_reset_reject(request_id):
    ensure_database_tables()
    rejected = update_password_reset_request_status(
        request_id,
        "Rejected",
        action_by=_safe_str(session.get("admin_email"), "").strip(),
    )
    if rejected:
        flash("Password reset request has been rejected.", "success")
    else:
        flash("Unable to reject this request. It may already be completed or unavailable.", "warning")
    return redirect(url_for("admin_password_reset_requests"))


@app.route("/admin/dataset", methods=["GET"])
@admin_required
def admin_dataset():
    ensure_database_tables()
    search_text = _safe_str(request.args.get("q"), "").strip()
    state_filter = _safe_str(request.args.get("state"), "all").strip()
    property_type_filter = _safe_str(request.args.get("property_type"), "all").strip()
    per_page_options = [25, 50, 75]
    try:
        page = parse_positive_int(request.args, "page", 1)
    except Exception:
        page = 1
    try:
        per_page = parse_positive_int(request.args, "per_page", per_page_options[0])
    except Exception:
        per_page = per_page_options[0]
    if per_page not in per_page_options:
        per_page = per_page_options[0]

    rows, use_mock = _fetch_admin_property_rows(
        search_text=search_text,
        state_filter=state_filter,
        property_type_filter=property_type_filter,
    )
    total_rows = len(rows)
    total_pages = max(1, (total_rows + per_page - 1) // per_page) if total_rows else 1
    if page > total_pages:
        page = total_pages
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paged_rows = rows[start_index:end_index]
    for item in paged_rows:
        property_id = _safe_int(item.get("property_id"), default=0)
        raw_image_url = _safe_str(item.get("image_url"), "").strip()
        item["image_url"] = _resolve_property_image_url_for_display(property_id, raw_image_url)

    page_window = 2
    page_start = max(1, page - page_window)
    page_end = min(total_pages, page + page_window)
    page_numbers = list(range(page_start, page_end + 1))
    pagination = {
        "page": page,
        "per_page": per_page,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "start_row": start_index + 1 if total_rows else 0,
        "end_row": min(end_index, total_rows),
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1,
        "next_page": page + 1,
        "page_numbers": page_numbers,
        "show_first_page": 1 not in page_numbers,
        "show_last_page": total_pages not in page_numbers,
    }

    return render_template(
        "admin/dataset.html",
        page_title="Property Dataset Management",
        active_admin_page="dataset",
        rows=paged_rows,
        filters={
            "q": search_text,
            "state": state_filter or "all",
            "property_type": property_type_filter or "all",
        },
        state_choices=get_state_choices(),
        property_type_options=PROPERTY_TYPE_OPTIONS,
        per_page_options=per_page_options,
        pagination=pagination,
        using_mock_data=use_mock,
    )


@app.route("/admin/dataset/new", methods=["GET", "POST"])
@admin_required
def admin_dataset_new():
    flash("Add Property has been disabled on admin page.", "info")
    return redirect(url_for("admin_dataset"))


@app.route("/admin/dataset/<int:property_id>", methods=["GET"])
@admin_required
def admin_dataset_view(property_id):
    ensure_database_tables()
    row = get_property_by_id(property_id)
    if not row:
        flash("Property record not found.", "warning")
        return redirect(url_for("admin_dataset"))
    image_url = _resolve_property_image_url_for_display(
        _safe_int(row.get("id"), default=0),
        _safe_str(row.get("image_url"), "").strip(),
    )
    return render_template(
        "admin/dataset_view.html",
        page_title="Property Dataset Details",
        active_admin_page="dataset",
        row=row,
        property_type_options=PROPERTY_TYPE_OPTIONS,
        furnishing_options=FURNISHING_OPTIONS,
        tenure_options=TENURE_OPTIONS,
        unit_type_options=UNIT_TYPE_OPTIONS,
        image_url=image_url,
    )


@app.route("/admin/dataset/<int:property_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_dataset_edit(property_id):
    ensure_database_tables()
    row = get_property_by_id(property_id)
    if not row:
        flash("Property record not found.", "warning")
        return redirect(url_for("admin_dataset"))
    if request.method == "POST":
        try:
            payload = _parse_admin_property_form(request.form)
            conn = get_db_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE property_listings
                        SET
                            title = %s,
                            area = %s,
                            negeri = %s,
                            property_type = %s,
                            built_up_sf = %s,
                            land_size = %s,
                            bedroom = %s,
                            bathroom = %s,
                            car_park = %s,
                            furnishing = %s,
                            tenure = %s,
                            unit_type = %s,
                            listing_price = %s,
                            latitude = %s,
                            longitude = %s,
                            image_url = %s
                        WHERE id = %s
                        """,
                        (
                            payload["title"],
                            payload["area"],
                            payload["negeri"],
                            payload["property_type"],
                            payload["built_up_sf"],
                            payload["land_size"],
                            payload["bedroom"],
                            payload["bathroom"],
                            payload["car_park"],
                            payload["furnishing"],
                            payload["tenure"],
                            payload["unit_type"],
                            payload["listing_price"],
                            payload["latitude"],
                            payload["longitude"],
                            payload["image_url"],
                            property_id,
                        ),
                    )
                flash("Property record updated successfully.", "success")
            finally:
                conn.close()
            return redirect(url_for("admin_dataset_view", property_id=property_id))
        except Exception as exc:
            flash(str(exc), "danger")
            row = dict(row)
            row.update(
                {
                    "title": _safe_str(request.form.get("title"), row.get("title")),
                    "area": _safe_str(request.form.get("area"), row.get("area")),
                    "negeri": _safe_str(request.form.get("negeri"), row.get("negeri")),
                    "image_url": _safe_str(request.form.get("image_url"), row.get("image_url")),
                }
            )
    edit_image_url = _safe_str(row.get("image_url"), "").strip() or None
    return render_template(
        "admin/dataset_form.html",
        page_title="Edit Property Record",
        active_admin_page="dataset",
        mode="edit",
        row=row,
        property_id=property_id,
        state_choices=get_state_choices(),
        property_type_options=PROPERTY_TYPE_OPTIONS,
        furnishing_options=FURNISHING_OPTIONS,
        tenure_options=TENURE_OPTIONS,
        unit_type_options=UNIT_TYPE_OPTIONS,
        image_url=edit_image_url,
    )


@app.route("/admin/dataset/<int:property_id>/delete", methods=["POST"])
@admin_required
def admin_dataset_delete(property_id):
    ensure_database_tables()
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM property_listings WHERE id = %s", (property_id,))
        flash("Property record deleted successfully.", "success")
    except Exception as exc:
        flash(f"Unable to delete property record: {exc}", "danger")
    finally:
        if conn:
            conn.close()
    return redirect(url_for("admin_dataset"))


@app.route("/admin/model-performance/rebuild", methods=["POST"])
@admin_required
def admin_model_rebuild():
    ensure_database_tables()
    try:
        summary = train_xgboost_after_tuned_model(test_size=0.2, random_state=42)
        flash(
            (
                "XGBoost after-tuning model has been rebuilt and saved successfully. "
                f"R2={summary['r2']:.4f}, RMSE={summary['rmse']:,.2f}, MAE={summary['mae']:,.2f}."
            ),
            "success",
        )
    except Exception as exc:
        flash(f"Unable to rebuild model: {exc}", "danger")
    return redirect(url_for("admin_model_performance"))


@app.route("/admin/model-performance", methods=["GET"])
@admin_required
def admin_model_performance():
    ensure_database_tables()
    get_model_quality_metrics.cache_clear()
    metrics = _get_admin_model_metrics()
    chart_data = {
        "labels": ["R2 Score", "RMSE", "MAE"],
        "series": [metrics["r2"], metrics["rmse"], metrics["mae"]],
    }
    return render_template(
        "admin/model_performance.html",
        page_title="Model Performance",
        active_admin_page="model-performance",
        metrics=metrics,
        chart_data=chart_data,
    )


@app.route("/admin/profile", methods=["GET", "POST"])
@admin_required
def admin_profile():
    if request.method == "POST":
        admin_name = _safe_str(request.form.get("admin_name"), "").strip() or ADMIN_DEFAULT_NAME
        admin_email = _safe_str(request.form.get("admin_email"), "").strip() or ADMIN_DEFAULT_EMAIL
        session["admin_name"] = admin_name
        session["admin_email"] = admin_email
        flash("Admin profile updated.", "success")
        return redirect(url_for("admin_profile"))
    profile = {
        "name": _safe_str(session.get("admin_name"), ADMIN_DEFAULT_NAME),
        "email": _safe_str(session.get("admin_email"), ADMIN_DEFAULT_EMAIL),
        "role": "System Administrator",
    }
    return render_template(
        "admin/profile.html",
        page_title="Admin Profile",
        active_admin_page="profile",
        profile=profile,
    )


if __name__ == "__main__":
    ensure_database_tables()
    app.run(debug=True, host="127.0.0.1", port=5000)
