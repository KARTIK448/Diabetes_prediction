import eventlet

eventlet.monkey_patch()

from flask import Flask, render_template, redirect, url_for, request, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import secrets
import hashlib
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired, Length
from flask_wtf.file import FileAllowed
from io import BytesIO
from flask_socketio import SocketIO, emit, join_room, leave_room
import tensorflow as tf
import numpy as np
from PIL import Image
import magic
from flask_cors import CORS
import gdown

gdrive_file_ids = {
    'vit_model.h5': '1IuXBRkdChnEaI1dEqUw4o3nH8nL7aTCW',
    'cnn_model.h5': '1XozLzTjlJib93A4Ac_d1-dllV4mkejzs',
    'resnet_model_improved.h5': '1VlWLnMgTO8-E2zsqzLHtmz34C0BiWah0'
}

model_path = 'models'  # updated to models folder


def download_model_from_gdrive(filename):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_path = os.path.join(model_path, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename} from Google Drive...")
        url = f"https://drive.google.com/uc?id={gdrive_file_ids[filename]}"
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"{filename} already exists locally.")


# Download all models if not present
for model_file in gdrive_file_ids.keys():
    download_model_from_gdrive(model_file)

class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

secret_key = secrets.token_hex(16)  # Generates a random 32-character hex string

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
# Update the database URI to use PostgreSQL. Replace with your actual credentials.
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

database_url = os.getenv('DATABASE_URL')
if not database_url:
    raise ValueError(
        "DATABASE_URL environment variable is not set. "
        "Please set it before running the application. "
        "Example: export DATABASE_URL=postgresql://user:password@host:port/database"
    )
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Models
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')
CORS(app)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'patient' or 'doctor'
    uploads = db.relationship('ImageUpload', backref='patient', lazy=True)  # Add this relationship


class ImageUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(150), nullable=False)
    image_data = db.Column(db.LargeBinary, nullable=False)  # Store image as BLOB
    result = db.Column(db.String(150), nullable=True)  # analysis result
    model_used = db.Column(db.String, nullable=True)  # model used for analysis
    prescriptions = db.relationship('Prescription', backref='image_upload', lazy=True)


from datetime import datetime
import pytz
import tzlocal

LOCAL_TIMEZONE = tzlocal.get_localzone()  # Automatically detect local timezone


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(pytz.utc))


@socketio.on('send_message')
def handle_send_message(data):
    room = get_room_id(data['doctor_id'], data['patient_id'])
    sender_id = data['sender_id']
    recipient_id = data['recipient_id']
    message_text = data['message']

    # Store the message in the database with UTC timestamp
    chat_msg = ChatMessage(sender_id=sender_id, recipient_id=recipient_id, message=message_text)
    db.session.add(chat_msg)
    db.session.commit()

    # Convert timestamp to local timezone string
    local_timestamp = chat_msg.timestamp.astimezone(LOCAL_TIMEZONE).strftime('%d-%m %H:%M')

    # Emit to all clients in the room, including timestamp in local time
    emit('receive_message', {
        'sender_id': sender_id,
        'message': message_text,
        'timestamp': local_timestamp
    }, room=room, include_self=False)  # change to True if you want sender to also get it


# Flask-Login user loader

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Routes

@app.route('/')
def home():
    return redirect(url_for('login'))


# Forms

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=4, max=150)])
    role = SelectField('Role', choices=[('patient', 'Patient'), ('doctor', 'Doctor')], validators=[DataRequired()])
    submit = SubmitField('Register')


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class UploadForm(FlaskForm):
    image = FileField('Upload Retinal Image',
                      validators=[DataRequired(), FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')])
    model_choice = SelectField('Select Model', choices=[
        ('vit_model.h5', 'ViT Model'),
        ('cnn_model.h5', 'CNN Model'),
        ('resnet_model_improved.h5', 'Resnet Model')
    ], validators=[DataRequired()])
    submit = SubmitField('Upload')


# Register route

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
        else:
            password_hash = hashlib.sha256(form.password.data.encode()).hexdigest()
            new_user = User(username=form.username.data, password=password_hash, role=form.role.data)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html', form=form)


# Login route

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        password_hash = hashlib.sha256(form.password.data.encode()).hexdigest()
        if user and user.password == password_hash:
            login_user(user)

            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html', form=form)


# Logout

@app.route('/logout')
@login_required
def logout():
    logout_user()

    return redirect(url_for('login'))


# Dashboard

@app.route('/dashboard')
@login_required
def dashboard():
    model_label_map = {
        'vit_model.h5': 'ViT Model',
        'cnn_model.h5': 'CNN Model',
        'resnet_model_improved.h5': 'Resnet Model'
    }
    if current_user.role == 'patient':
        uploads = ImageUpload.query.filter_by(user_id=current_user.id).all()
        doctors = User.query.filter_by(role='doctor').all()
        return render_template('patient_dashboard.html', uploads=uploads, doctors=doctors,
                               model_label_map=model_label_map)
    else:
        # doctor sees list of patients
        patients = User.query.filter_by(role='patient').all()
        return render_template('doctor_dashboard.html', patients=patients, model_label_map=model_label_map)


# Upload image route


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if current_user.role != 'patient':
        flash('Only patients can upload images.', 'danger')
        return redirect(url_for('dashboard'))

    form = UploadForm()
    if form.validate_on_submit():
        file = form.image.data
        filename = secure_filename(file.filename)
        selected_model_file = form.model_choice.data
        original_image_data = file.read()
        # Process image bytes to model input
        img = Image.open(BytesIO(original_image_data)).convert('RGB')
        img = img.resize((224, 224))
        # Convert processed image back to bytes
        processed_img_io = BytesIO()
        img.save(processed_img_io, format='PNG')
        processed_img_io.seek(0)
        processed_image_data = processed_img_io.read()
        img_array = np.array(img) / 255.0  # normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension

        # Dynamically load the selected model
        model_file_path = os.path.join(model_path, selected_model_file)
        selected_model = tf.keras.models.load_model(model_file_path)

        # Predict using selected model
        prediction = selected_model.predict(img_array)
        pred_class_idx = np.argmax(prediction)
        result = class_names[pred_class_idx]

        # Save the processed image bytes, result, and model used to DB
        new_image = ImageUpload(user_id=current_user.id, filename=filename, image_data=processed_image_data,
                                result=result, model_used=selected_model_file)
        db.session.add(new_image)
        db.session.commit()
        flash('Image uploaded and saved to database.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('upload.html', form=form)


# Route to view image from DB

@app.route('/image/<int:image_id>')
@login_required
def view_image(image_id):
    image = ImageUpload.query.get_or_404(image_id)
    img_bytes = image.image_data
    # Use python-magic to determine MIME type
    mime_type = magic.from_buffer(img_bytes, mime=True)
    return send_file(BytesIO(img_bytes), download_name=image.filename, mimetype=mime_type)


@app.route('/patient/<int:patient_id>')
@login_required
def patient_profile(patient_id):
    if current_user.role != 'doctor':
        flash('Only doctors can view patient profiles.', 'danger')
        return redirect(url_for('dashboard'))
    patient = User.query.get_or_404(patient_id)
    uploads = ImageUpload.query.filter_by(user_id=patient.id).all()
    return render_template('patient_profile.html', patient=patient, uploads=uploads)


@app.route('/chat/<int:patient_id>/<int:doctor_id>')
@login_required
def chat(patient_id, doctor_id):
    try:
        # Load users from DB
        doctor = User.query.get_or_404(doctor_id)
        patient = User.query.get_or_404(patient_id)

        # Ensure roles are correct
        if doctor.role != 'doctor' or patient.role != 'patient':
            flash("Invalid chat combination. Must be between a doctor and a patient.", "danger")
            return redirect(url_for('dashboard'))

        # Ensure the logged-in user is a participant
        if current_user.id not in {doctor.id, patient.id}:
            flash("Unauthorized chat access.", "danger")
            return redirect(url_for('dashboard'))

        # Fetch chat messages
        messages = ChatMessage.query.filter(
            ((ChatMessage.sender_id == doctor.id) & (ChatMessage.recipient_id == patient.id)) |
            ((ChatMessage.sender_id == patient.id) & (ChatMessage.recipient_id == doctor.id))
        ).order_by(ChatMessage.timestamp.asc()).all()

        message_history = [
            {
                'sender_id': msg.sender_id,
                'sender_role': User.query.get(msg.sender_id).role,
                'message': msg.message,
                'timestamp': msg.timestamp.strftime('%d-%m %H:%M') if msg.timestamp else None
            } for msg in messages
        ]

        chat_with = patient.username if current_user.role == 'doctor' else doctor.username
        response = render_template(
            'chat.html',
            doctor=doctor,
            patient=patient,
            current_user=current_user,
            page_title=f"Chat with {chat_with}",
            message_history=message_history
        )
        headers = {
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        return response, 200, headers

    except Exception as e:
        import traceback
        print("[ERROR] Chat route failed:")
        traceback.print_exc()
        flash("An error occurred while opening the chat.", "danger")
        return redirect(url_for('dashboard'))



    except Exception as e:
        print(f"[Error] Chat route failed: {e}")
        flash("An error occurred while opening the chat.", "danger")
        return redirect(url_for('dashboard'))


@socketio.on('join')
def on_join(data):
    room = get_room_id(data['doctor_id'], data['patient_id'])
    join_room(room)


@socketio.on('leave')
def on_leave(data):
    room = get_room_id(data['doctor_id'], data['patient_id'])
    leave_room(room)


def get_room_id(doctor_id, patient_id):
    # Create a consistent room id string
    ids = sorted([doctor_id, patient_id])
    return f'chat_{ids[0]}_{ids[1]}'


# Create tables in the database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)

# For Gunicorn compatibility
def create_app():
    return app
