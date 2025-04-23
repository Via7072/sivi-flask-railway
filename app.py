from flask import Flask, request, jsonify, session
from flask_bcrypt import Bcrypt #pip install Flask-Bcrypt = https://pypi.org/project/Flask-Bcrypt/
import cv2
import numpy as np
import base64
import onnxruntime as ort
from flask_cors import CORS, cross_origin #ModuleNotFoundError: No module named 'flask_cors' = pip install Flask-Cors
from models import db, User
from flask_mail import Mail, Message
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')

mail = Mail(app)

# Load ONNX model
session_onnx = ort.InferenceSession("best.onnx")

# Class names
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", 
               "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", 
               "U", "V", "W", "X", "Y", "Z"]  

CONFIDENCE_THRESHOLD = 0.5

SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True

bcrypt = Bcrypt(app)
db.init_app(app)

with app.app_context():
    db.create_all()

def send_payment_email(email):
    try:
        msg = Message('Instruksi Pembayaran', recipients=[email])
        msg.body = f"""
        Halo,

        Terima kasih telah mendaftar. Untuk mengaktifkan akun Anda, silakan lakukan pembayaran sesuai instruksi berikut:

        - Nomor Rekening: 1234567890 (BCA)
        - Nama Penerima: Nama Anda
        - Jumlah: Rp 50.000 (contoh)
        - Kirim bukti transfer ke email ini.

        Setelah pembayaran dikonfirmasi, akun Anda akan diaktifkan.

        Terima kasih!
        """
        mail.send(msg)
        print(f"Email terkirim ke {email}")
    except Exception as e:
        print(f"Error mengirim email: {str(e)}")

@app.route("/")
def home():
    return "Welcome to SIVI server with Flask!"

@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    users_data = [{
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_paid": user.is_paid,
        "role": user.role
    } for user in users]
    
    return jsonify(users_data), 200

@app.route("/signup", methods=["POST"])
def signup():
    username = request.json["username"]
    email = request.json["email"]
    password = request.json["password"]
    is_paid = request.json["is_paid"]
    role = request.json.get("role", "user")

    user_exists = User.query.filter_by(email=email).first() is not None

    if user_exists:
        return jsonify({"error": "Email already exists"}), 409
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, email=email, password=hashed_password, is_paid=is_paid, role=role)
    db.session.add(new_user)
    db.session.commit()

    send_payment_email(email)

    session["user_id"] = new_user.id

    return jsonify({
        "id": new_user.id,
        "username": new_user.username,
        "email": new_user.email,
        "is_paid": new_user.is_paid,
        "role": new_user.role
    })

@app.route("/login", methods=["POST"])
def login_user():
    email = request.json["email"]
    password = request.json["password"]

    user = User.query.filter_by(email=email).first()

    if user is None:
        return jsonify({"error": "unauthorized"}), 401
    
    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Unauthorized"}), 401
    
    if not user.is_paid:
        return jsonify({"error": "Payment required"}), 402
    
    session["user_id"] = user.id

    return jsonify({
        "id": user.id,
        "email": user.email,
        "is_paid": user.is_paid,
        "role": user.role
    })

@app.route('/update-user-status', methods=['POST'])
def update_user_status():
    data = request.get_json()
    email = data.get('email')

    user = User.query.filter_by(email=email).first()

    if user:
        user.is_paid = True
        db.session.commit()
        return jsonify({"is_paid": user.is_paid}), 200
    else:
        return jsonify({"error": "Email not found"}), 404
    
@app.route('/update-role', methods=['POST'])
def update_role():
    data = request.get_json()
    email = data.get('email')

    user = User.query.filter_by(email=email).first()

    if user:
        user.role = "admin"
        db.session.commit()
        return jsonify({"role": user.role}), 201
    else:
        return jsonify({"error": "Email not found"}), 404
    
@app.route('/delete-user', methods=['POST'])
def delete_user():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({"error": "Email tidak boleh kosong"}), 400

    user = User.query.filter_by(email=email).first()

    if user:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": f"User dengan email {email} berhasil dihapus."}), 200
    else:
        return jsonify({"error": "User tidak ditemukan"}), 404

@app.route('/process_frame', methods=['POST'])
@cross_origin()
def process_frame():
    try:
        total_start_time = time.time()  # Mulai hitung waktu total

        # 1️⃣ Terima data dari request
        start_time = time.time()
        data = request.json
        frame_base64 = data.get("frame")
        if not frame_base64:
            return jsonify({"error": "No frame provided"}), 400
        print(f"🕒 Waktu terima request: {(time.time() - start_time) * 1000:.2f} ms")

        # 2️⃣ Decode base64 image
        start_time = time.time()
        frame_data = base64.b64decode(frame_base64)
        print(f"🕒 Waktu decode base64: {(time.time() - start_time) * 1000:.2f} ms")

        # 3️⃣ Convert to numpy array
        start_time = time.time()
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
        print(f"🖼️ Ukuran gambar yang diproses: {frame.shape}")  # (tinggi, lebar, jumlah_channel)
        print(f"🕒 Waktu konversi ke numpy array: {(time.time() - start_time) * 1000:.2f} ms")

        # 4️⃣ Preprocess image (resize, etc.)
        start_time = time.time()
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (640, 640))
        input_tensor = input_image.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        print(f"🕒 Waktu preprocessing gambar: {(time.time() - start_time) * 1000:.2f} ms")

        # 5️⃣ Run inference with ONNX model
        start_time = time.time()
        inputs = {session_onnx.get_inputs()[0].name: input_tensor}
        outputs = session_onnx.run(None, inputs)
        print(f"🕒 Waktu inferensi ONNX: {(time.time() - start_time) * 1000:.2f} ms")

        # 6️⃣ Process output
        start_time = time.time()
        output = outputs[0]
        boxes = output[0, :, :4]
        probabilities = output[0, :, 4]
        class_ids = output[0, :, 5].astype(int)

        height, width, _ = frame.shape
        detections = []
        indices = np.where(probabilities > CONFIDENCE_THRESHOLD)[0]

        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            x1 = int(x1 * (width / 640))
            y1 = int(y1 * (height / 640))
            x2 = int(x2 * (width / 640))
            y2 = int(y2 * (height / 640))
            class_id = class_ids[i]
            detections.append({
                "class": class_names[class_id],
                "confidence": round(float(probabilities[i]), 2),
                "bbox": [x1, y1, x2, y2]
            })
        print(f"🕒 Waktu parsing hasil deteksi: {(time.time() - start_time) * 1000:.2f} ms")

        # 7️⃣ Total processing time
        total_processing_time = (time.time() - total_start_time) * 1000
        print(f"✅ Total waktu pemrosesan: {total_processing_time:.2f} ms")

        return jsonify(detections)
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)