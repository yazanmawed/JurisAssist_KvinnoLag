from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import smtplib
from email.mime.text import MIMEText



app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
CORS(app)



# DON'T forget its for test purposes only change based on company mail:
# Configuration for your SMTP server
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'university.west.campus.link@gmail.com' #from email
SMTP_PASSWORD = 'hmuy zint phob fbdj'                   # password token for the email
TO_EMAIL = 'university.west.campus.link@gmail.com'      # The email address you want to send to "kontakta@kvinnolag.se"


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
        (["hej", "tja", "hejsan", "tjena", "hello"], f"Hej {user.first_name}! Hur kan jag hjälpa dig idag?"),
        (["gm", "gm!", "godmorgon"], f"Godmorgon {user.first_name}! Hur kan jag hjälpa dig idag?"),
        (["gd", "gd!", "goddag"], f"Goddag {user.first_name}! Hur kan jag hjälpa dig idag?"),
        (["hjälp", "help"], "Jag är här för att hjälpa dig. Var vänlig beskriv ditt problem."),
        (["hejdå", "bye", "adjö"], "Hejdå! Tack för att du kontaktade oss."),
        (["ha det så bra"], "Tack! det samma")
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

@app.route('/sendmail', methods=['POST'])    
def send_mail():
    namn = request.form.get('namn')
    from_email = request.form.get('email')
    ämne = request.form.get('ämne')
    meddelande = request.form.get('meddelande')

    # ---------------------------
    # Email #1: Send to company
    # ---------------------------
    body_company = f"""
    Namn: {namn}
    E-post: {from_email}
    Ämne: {ämne}

    Meddelande:
    {meddelande}
    """
    msg_company = MIMEText(body_company)
    msg_company['Subject'] = f"Kontaktformulär: {ämne}"
    msg_company['From'] = SMTP_USERNAME
    msg_company['To'] = TO_EMAIL
    msg_company.add_header('Reply-To', from_email)

    # ---------------------------
    # Email #2: Send confirmation to user
    # ---------------------------
    body_user = f"""
    Hej {namn},

    Tack för ditt meddelande om "{ämne}".
    Vi återkommer så snart som möjligt.

    Med Vänliga Hälsningar,
    KVINNOLAG Juristbyrå
    """
    msg_user = MIMEText(body_user)
    msg_user['Subject'] = f"Bekräftelse: {ämne}"
    msg_user['From'] = SMTP_USERNAME
    msg_user['To'] = from_email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)

            # Send email to company
            server.sendmail(msg_company['From'], msg_company['To'], msg_company.as_string())
            
            # Send confirmation email to the user
            server.sendmail(msg_user['From'], msg_user['To'], msg_user.as_string())

        return jsonify({"status": "success", "message": "Meddelande skickat!"})
    
    except Exception as e:
        print(f"Error sending emails: {e}")
        return jsonify({"status": "error", "message": "Något gick fel."}), 500

@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
