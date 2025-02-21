from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key_here'  # Required for session

# Temporary storage (replace with database in production)
users = {}
user_counter = 1

class User:
    def __init__(self, first_name, last_name, age, gender, email, problem=None):
        global user_counter
        self.id = user_counter
        user_counter += 1
        self.first_name = first_name
        self.last_name = last_name
        self.age = age
        self.gender = gender
        self.email = email
        self.problem = problem

def ai_chatbot_response(user_message, user):
    responses = [
        (["hej", "tja", "hejsan", "hello"], f"Hej {user.first_name}! Hur kan jag hjälpa dig idag?"),
        (["hjälp", "help"], "Jag är här för att hjälpa dig. Var vänlig beskriv ditt problem."),
        (["hejdå", "adjö"], "Hejdå! Tack för att du kontaktade oss.")
    ]
    
    lower_msg = user_message.lower()
    for triggers, response in responses:
        if lower_msg in triggers:
            return response
    return "Jag förstår inte riktigt. Kan du vara snäll och förtydliga din fråga?"

@app.route('/create_user', methods=['POST'])
def create_user():
    global users
    data = request.get_json()
    user = User(
        data['first_name'],
        data['last_name'],
        data['age'],
        data['gender'],
        data['email'],
        #data['problem']
    )
    users[user.id] = user
    session['user_id'] = user.id
    return jsonify({'user_id': user.id})

@app.route('/get_response', methods=['POST'])
def get_ai_response():
    data = request.get_json()
    user_message = data.get('message', '')
    user_id = session.get('user_id')
    
    if not user_id or user_id not in users:
        return jsonify({'error': 'User not registered'}), 400
    
    user = users[user_id]
    response = ai_chatbot_response(user_message, user)
    return jsonify({
        'response': response,
        'user_name': user.first_name
    })


@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
