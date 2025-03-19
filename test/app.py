from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import smtplib
from email.mime.text import MIMEText
import os
import pickle
import numpy as np
from tensorflow import keras
from legal_bot import LegalChatbot

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
CORS(app)

# Initialize the chatbot
chatbot = LegalChatbot()

# Train and load model if not already trained
MODEL_FILE = 'legal_chatbot_model.h5'
DATA_FILE = 'legal_chatbot_data.pkl'

# SMTP Configuration (For sending emails)
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'university.west.campus.link@gmail.com'  # From email
SMTP_PASSWORD = 'hmuy zint phob fbdj'  # App password token for the email
TO_EMAIL = 'university.west.campus.link@gmail.com'  # Default recipient


def send_email(to_email, subject, body):
    """Sends an email notification to the user"""
    try:
        print(f"📩 Attempting to send email to: {to_email}")
        print(f"📧 Email Subject: {subject}")

        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = SMTP_USERNAME
        msg['To'] = to_email

        # Establish a connection with the SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(SMTP_USERNAME, SMTP_PASSWORD)  # Login to the SMTP server

        server.sendmail(SMTP_USERNAME, to_email, msg.as_string())  # Send the email
        server.quit()  # Close the connection

        print("✅ Email sent successfully to", to_email)
        return True
    except smtplib.SMTPAuthenticationError:
        print("❌ SMTP Authentication Error: Incorrect username/password or Google is blocking sign-in.")
    except smtplib.SMTPConnectError:
        print("❌ SMTP Connection Error: Unable to connect to the server.")
    except smtplib.SMTPException as e:
        print(f"❌ SMTP Error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")

    return False


if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
    chatbot.build_model()  # Train model if not found

# Load trained model
chatbot.model = keras.models.load_model(MODEL_FILE)
with open(DATA_FILE, 'rb') as f:
    data = pickle.load(f)
    chatbot.words = data['words']
    chatbot.classes = data['classes']

# Temporary storage for users
users = {}
user_counter = 1


class User:
    """Class representing a user"""
    def __init__(self, first_name, last_name, age, gender, email):
        global user_counter
        self.id = user_counter
        user_counter += 1
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.gender = gender
        self.email = email


def get_chatbot_response(message, user_id):
    """Fetches the chatbot's response for a given user"""
    if user_id not in users:
        return "User not found. Please register first."

    chatbot_response = chatbot.handle_conversation(message, session_id=str(user_id))

    if chatbot_response is None:
        chatbot_response = ("I'm sorry, but I couldn't understand that. Can you rephrase?", None)

    response, pricing_info = chatbot_response
    return response if not pricing_info else f"{response}\n\n{pricing_info}"


@app.route('/create_user', methods=['POST'])
def create_user():
    """Handles user registration"""
    global users
    data = request.get_json()
    user = User(
        data['first_name'],
        data['last_name'],
        data['age'],
        data['gender'],
        data['email']
    )
    users[user.id] = user
    session['user_id'] = user.id
    session['user_email'] = user.email  # Store user email for appointment confirmation
    print(f"📧 User {user.first_name} registered with email: {user.email}")
    return jsonify({'user_id': user.id})


@app.route('/get_response', methods=['POST'])
def get_ai_response():
    data = request.get_json()
    user_message = data.get('message', '')
    user_id = session.get('user_id')

    print(f"📥 Received message: {user_message}")

    if not user_id or user_id not in users:
        print("⚠️ User not registered.")
        return jsonify({'error': 'User not registered'}), 400

    try:
        response, metadata = chatbot.handle_conversation(user_message, str(user_id))
        
        print(f"🤖 Chatbot Response: {response}")
        print(f"📊 Metadata: {metadata}")

        return jsonify({'response': response, 'user_name': users[user_id].first_name})
    except Exception as e:
        print(f"❌ Error in Flask: {e}")
        return jsonify({'error': str(e)}), 500


    
    
@app.route('/')
def index():
    """Renders the homepage"""
    return render_template('index.html')


def send_email(to_email, subject, body):
    try:
        print(f"📩 Attempting to send email to: {to_email}")
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = SMTP_USERNAME
        msg['To'] = to_email

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.set_debuglevel(1)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, to_email, msg.as_string())
        server.quit()
        print("✅ Email sent successfully")
        return True
    except Exception as e:
        print(f"❌ Email failed to send: {str(e)}")
        return False



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
