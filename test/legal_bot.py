import json
import re
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta




# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Create a more comprehensive knowledge base
knowledge_base = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hello", "Hi", "Hey", "Good morning", "Good afternoon", 
                "Good evening", "Hej", "Hejsan", "God morgon", "God kväll"
            ],
            "responses": [
                "Hello! How can I assist you with your legal query today?",
                "Hi there! I'm the virtual assistant for Kvinnolag Juristbyrå. What legal matter can I help you with?",
                "Welcome to Kvinnolag Juristbyrå! How may I assist you with your legal concerns?"
            ]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "Bye", "Goodbye", "See you later", "Take care", "Hejdå", "Adjö", "Vi ses"
            ],
            "responses": [
                "Goodbye! If you have more questions later, feel free to ask.",
                "Thank you for contacting Kvinnolag Juristbyrå. Have a great day!",
                "Thank you for your time. Feel free to reach out if you need further assistance."
            ]
        },
        {
            "tag": "thanks",
            "patterns": [
                "Thank you", "Thanks", "Thanks a lot", "I appreciate your help", "Tack", "Tack så mycket"
            ],
            "responses": [
                "You're welcome! Is there anything else I can assist you with?",
                "Happy to help! Do you have any other legal questions?",
                "It's my pleasure to assist. Is there anything else you need help with?"
            ]
        },
        {
            "tag": "name",
            "patterns": [
                "What is your name?", "Who are you?", "What should I call you?", "Vad heter du?", "Vem är du?"
            ],
            "responses": [
                "I am the virtual legal assistant for Kvinnolag Juristbyrå.",
                "I'm an automated assistant designed to help with legal queries at Kvinnolag Juristbyrå.",
                "You can call me Kvinnolag's legal assistant. How can I help you today?"
            ]
        },
        {
            "tag": "hours",
            "patterns": [
                "What are your hours?", "When are you open?", "Working hours", "Office hours",
                "När har ni öppet?", "Öppettider"
            ],
            "responses": [
                "Kvinnolag Juristbyrå is open Monday to Friday from 9:00 AM to 5:00 PM. Would you like to schedule a consultation?"
            ]
        },
        {
            "tag": "contact",
            "patterns": [
                "How can I contact you?", "Phone number", "Email address", "Contact information", "Office location",
                "Hur kan jag kontakta er?", "Telefonnummer", "E-postadress", "Kontaktinformation", "Kontorets plats"
            ],
            "responses": [
                "You can contact Kvinnolag Juristbyrå by phone at +46-XX-XXX-XXXX or by email at info@kvinnolag.se. Our office is located in Stockholm."
            ]
        },
        {
            "tag": "divorce",
            "category": "family",
            "patterns": [
                "How much does a divorce cost?", "I need help with divorce", "Divorce proceedings", "Legal fees for divorce",
                "Uncontested divorce", "Contested divorce", "Separation", "Child custody during divorce",
                "Hur mycket kostar en skilsmässa?", "Jag behöver hjälp med skilsmässa", "Skilsmässoförfarande"
            ],
            "responses": [
                "For divorce cases, the cost typically ranges from 500kr to 1500kr for an uncontested divorce. Contested divorces with issues like property division or child custody can range from 2000kr to 10000kr depending on complexity. Would you like to schedule a consultation with one of our family law specialists?"
            ],
            "pricing": {
                "uncontested": "500-1500kr",
                "contested": "2000-10000kr",
                "consultation": "Free for first 30 minutes"
            }
        },
        {
            "tag": "child_custody",
            "category": "family",
            "patterns": [
                "Child custody issues", "Joint custody", "Sole custody", "Custody battle", "Parenting plan",
                "Child visitation rights", "Child support", "Modify custody agreement",
                "Vårdnadstvist", "Gemensam vårdnad", "Ensam vårdnad", "Umgängesrätt", "Underhållsbidrag"
            ],
            "responses": [
                "Child custody cases typically involve determining parental rights, visitation schedules, and child support. The cost varies widely based on complexity, ranging from 3000kr to 15000kr. We offer personalized guidance for your specific situation. Would you like to schedule a consultation with our family law specialist?"
            ],
            "pricing": {
                "simple": "3000-5000kr",
                "complex": "5000-15000kr",
                "consultation": "Free for first 30 minutes"
            }
        },
        {
            "tag": "contracts",
            "category": "contract",
            "patterns": [
                "Contract review", "Draft a contract", "Contract terms", "Legal agreement", "Business contract",
                "Rental agreement", "Employment contract", "Contract dispute", "Contract breach",
                "Kontraktsgranskning", "Upprätta ett kontrakt", "Avtalsvillkor", "Juridiskt avtal", "Affärsavtal"
            ],
            "responses": [
                "For contract-related services, we offer drafting, review, and dispute resolution. Basic contract reviews start at 1000kr, while contract drafting ranges from 2000kr to 5000kr depending on complexity. Contract disputes vary based on the issue. Would you like more specific information about your contract needs?"
            ],
            "pricing": {
                "review": "1000-3000kr",
                "drafting": "2000-5000kr",
                "dispute": "3000-10000kr",
                "consultation": "Free for first 30 minutes"
            }
        },
        {
            "tag": "labor",
            "category": "labor",
            "patterns": [
                "Workplace discrimination", "Wrongful termination", "Employment law", "Worker's rights",
                "Hostile work environment", "Sexual harassment", "Unpaid wages", "Overtime pay",
                "Labor dispute", "Employment contract", "Severance package", "Non-compete agreement",
                "Arbetsdiskriminering", "Felaktig uppsägning", "Arbetsrätt", "Arbetstagarens rättigheter"
            ],
            "responses": [
                "For labor law matters, we provide representation and advice on workplace issues including discrimination, wrongful termination, and contract disputes. Initial consultations cost 750kr, with case representation ranging from 2500kr to 8000kr depending on complexity. Would you like to discuss your specific employment situation?"
            ],
            "pricing": {
                "consultation": "750kr",
                "simple_case": "2500-4000kr",
                "complex_case": "4000-8000kr"
            }
        },
        {
            "tag": "property",
            "category": "property",
            "patterns": [
                "Property dispute", "Real estate law", "Property rights", "Boundary dispute",
                "Zoning laws", "Land use", "Easement", "Property title", "Property transfer",
                "Landlord issues", "Tenant problems", "Eviction", "Lease agreement",
                "Fastighetstvist", "Fastighetsrätt", "Äganderätt", "Gränstvist"
            ],
            "responses": [
                "Our property law services cover disputes, transactions, and landlord-tenant issues. Property document preparation ranges from 1500kr to 3000kr, while representation in disputes starts at 3000kr. Would you like specific information about your property concern?"
            ],
            "pricing": {
                "document_preparation": "1500-3000kr",
                "dispute_representation": "3000-7500kr",
                "consultation": "Free for first 30 minutes"
            }
        },
        {
            "tag": "immigration",
            "category": "immigration",
            "patterns": [
                "Visa application", "Residency permit", "Citizenship", "Asylum", "Immigration process",
                "Work permit", "Student visa", "Family reunification", "Deportation", "Appeal immigration decision",
                "Visumansökan", "Uppehållstillstånd", "Medborgarskap", "Asyl", "Invandringsprocess"
            ],
            "responses": [
                "For immigration matters, we assist with visa applications, residency permits, citizenship, and appeals. Basic application assistance starts at 2000kr, with more complex cases ranging from 3500kr to 7000kr. Would you like to discuss your immigration status in detail?"
            ],
            "pricing": {
                "basic_assistance": "2000-3500kr",
                "complex_case": "3500-7000kr",
                "appeals": "4000-8000kr",
                "consultation": "750kr"
            }
        },
        {
            "tag": "criminal",
            "category": "criminal",
            "patterns": [
                "Criminal defense", "Criminal charges", "Arrest", "Police questioning", "Criminal record",
                "DUI", "Theft charges", "Assault charges", "Drug charges", "Court representation",
                "Brottmålsförsvar", "Brottsanklagelser", "Arrestering", "Polisförhör", "Belastningsregister"
            ],
            "responses": [
                "For criminal law matters, we provide legal defense, advice during police questioning, and court representation. Consultation fees start at 1000kr, with defense services ranging from 5000kr to 20000kr based on case complexity. Given the serious nature of criminal cases, we recommend an immediate consultation to discuss your situation."
            ],
            "pricing": {
                "consultation": "1000kr",
                "minor_offense": "5000-10000kr",
                "major_offense": "10000-20000kr"
            }
        },
        {
            "tag": "appointment",
            "patterns": [
                "Schedule appointment", "Book consultation", "Meet with lawyer", "Legal consultation",
                "Free consultation", "Book meeting", "Available times", "Appointment slots",
                "Boka tid", "Boka konsultation", "Träffa advokat", "Juridisk konsultation"
            ],
            "responses": [
                "I'd be happy to help you schedule a consultation with one of our lawyers. We offer a free 30-minute initial consultation for most legal matters. What type of legal issue would you like to discuss, and what days/times work best for you?"
            ]
        },
        {
            "tag": "gdpr",
            "patterns": [
                "Data protection", "Privacy policy", "GDPR", "Data rights", "Personal information",
                "Data storage", "Delete my data", "Privacy concerns", "Information storage",
                "Dataskydd", "Integritetspolicy", "Datarättigheter", "Personlig information"
            ],
            "responses": [
                "Kvinnolag Juristbyrå takes data protection seriously. In accordance with GDPR, we only collect and process necessary information with your consent. You can request access to, modification of, or deletion of your personal data at any time. Would you like more information about our privacy practices?"
            ]
        },
        {
            "tag": "fallback",
            "patterns": [],
            "responses": [
                "I'm not sure I understand your query. Could you please rephrase or specify which legal area you're interested in?",
                "I don't have enough information to answer that question. Could you provide more details about your legal concern?",
                "I'm still learning about specific legal matters. Would you like to schedule a consultation with one of our lawyers to discuss this in detail?"
            ]
        }
    ]
}





class LegalChatbot:
    def __init__(self):
        self.knowledge_base = knowledge_base
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!', '.', ',', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', "'", '"']
        self.model = None
        self.conversation_state = {}
        self.initialize_data()
        self.build_model()
        
    def initialize_data(self):
        # Process the intents and create training data
        for intent in self.knowledge_base["intents"]:
            for pattern in intent["patterns"]:
                # Tokenize each word in the pattern
                w = nltk.word_tokenize(pattern)
                # Add to our words list
                self.words.extend(w)
                # Add to documents in our corpus
                self.documents.append((w, intent["tag"]))
                # Add to our classes list
                if intent["tag"] not in self.classes:
                    self.classes.append(intent["tag"])

        # Lemmatize, lowercase and remove duplicates
        self.words = [lemmatizer.lemmatize(word.lower()) for word in self.words if word not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        print(f"Unique lemmatized words: {len(self.words)}")
        print(f"Unique intent classes: {len(self.classes)}")
        print(f"Training documents: {len(self.documents)}")
        
    def create_training_data(self):
        training = []
        output_empty = [0] * len(self.classes)
        
        for doc in self.documents:
            # Initialize our bag of words
            bag = []
            # List of tokenized words for the pattern
            pattern_words = doc[0]
            # Lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            # Create our bag of words array
            for word in self.words:
                bag.append(1) if word in pattern_words else bag.append(0)
                
            # Output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
            
        # Shuffle features and make numpy array
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        # Create train and test lists. X - patterns, Y - intents
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        return train_x, train_y
    
    def build_model(self):
        # Create model - 3 layers
        # First layer - 128 neurons, second layer - 64 neurons, 3rd output layer contains number of neurons
        # equal to number of intents to predict output intent with softmax
        
        train_x, train_y = self.create_training_data()
        
        model = keras.Sequential([
            keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(train_y[0]), activation='softmax')
        ])
        
        # Compile model
        model.compile(loss='categorical_crossentropy', 
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      metrics=['accuracy'])
        
        # Fitting and saving the model
        history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        
        self.model = model
        
        # Save model, words, and classes
        model.save('legal_chatbot_model.h5')
        pickle.dump({'words': self.words, 'classes': self.classes, 'train_x': train_x, 'train_y': train_y}, 
                    open('legal_chatbot_data.pkl', 'wb'))
        
        print("Model built and saved")
        
    def clean_up_sentence(self, sentence):
        # Tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # Lemmatize each word - create base form, to try to represent related words
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence):
        # Tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # Bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    # Assign 1 if current word is in the vocabulary position
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        # Filter out predictions below a threshold
        bow = self.bow(sentence)
        
        if sum(bow) == 0:  # If no words match our vocabulary
            return [{'intent': 'fallback', 'probability': 1.0}]
            
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        # Sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, intent_list, knowledge_base):
        if not intent_list:
            tag = 'fallback'
        else:
            tag = intent_list[0]['intent']
        
        # Find the corresponding intent in the knowledge base
        for intent in knowledge_base['intents']:
            if intent['tag'] == tag:
                # Return a random response from the intent
                return random.choice(intent['responses']), tag, intent.get('pricing', None)
                
        # If no matching intent is found, return a fallback response
        for intent in knowledge_base['intents']:
            if intent['tag'] == 'fallback':
                return random.choice(intent['responses']), 'fallback', None

    def generate_pricing_info(self, tag, pricing_data):
        if not pricing_data:
            return "I don't have specific pricing information for this legal matter. Would you like to schedule a consultation to discuss costs?"
        
        pricing_info = "Here's a pricing estimate for this legal service:\n"
        
        for service_type, price in pricing_data.items():
            service_name = service_type.replace('_', ' ').capitalize()
            pricing_info += f"- {service_name}: {price}\n"
            
        pricing_info += "\nPlease note that actual costs may vary based on the specific details of your case. Would you like to schedule a consultation for a more accurate quote?"
        
        return pricing_info
    
    def schedule_appointment(self, legal_area=None):
        # Simulate appointment scheduling
        available_slots = []
        today = datetime.now()
        
        # Generate some fictional available time slots over the next 5 days
        for i in range(1, 6):
            day = today + timedelta(days=i)
            # Skip weekends
            if day.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                continue
            
            # Create slots at 9:00, 11:00, 13:00 and 15:00
            for hour in [9, 11, 13, 15]:
                slot = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                available_slots.append(slot)
        
        # Format the available slots for display
        formatted_slots = []
        for slot in available_slots:
            formatted_slots.append(slot.strftime("%A, %B %d at %H:%M"))
        
        # Create a response with available appointment times
        response = "I can help you schedule a 30-minute consultation with one of our lawyers. "
        
        if legal_area:
            response += f"For your {legal_area} matter, "
        
        response += "here are some available time slots:\n\n"
        
        for i, slot in enumerate(formatted_slots[:5]):  # Show only 5 slots
            response += f"{i+1}. {slot}\n"
        
        response += "\nTo book an appointment, please let me know which time slot works for you, or contact our office directly at +46-XX-XXX-XXXX."
        
        return response
    
    def handle_conversation(self, user_input, session_id="default"):
        """Handles user conversation and tracks session state."""
        
        # Initialize conversation state if it does not exist
        if session_id not in self.conversation_state:
            self.conversation_state[session_id] = {
                "context": None,
                "followup": None,
                "legal_area": None
            }

        state = self.conversation_state[session_id]

        # ✅ Handling Appointment Scheduling
        if state["followup"] == "appointment_scheduling":
            # ✅ If user simply says "yes", provide available slots
            if user_input.lower() in ["yes", "yeah", "sure", "y", "ok", "of course"]:
                return self.schedule_appointment(state["legal_area"]), None
            
            # ✅ Generate appointment slots dynamically
            available_slots = []
            today = datetime.now()
            
            for i in range(1, 6):  # Generate slots for the next 5 weekdays
                day = today + timedelta(days=i)
                if day.weekday() < 5:  # Skip weekends (Saturday = 5, Sunday = 6)
                    for hour in [9, 11, 13, 15]:  # Office hours
                        slot = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                        available_slots.append(slot.strftime("%A, %B %d at %H:%M"))

            # ✅ Check if user's input matches an available slot
            selected_slot = next((slot for slot in available_slots if slot.lower() in user_input.lower()), None)

            if selected_slot:
                state["followup"] = None
                confirmation_msg = f"✅ Your appointment for {selected_slot} has been scheduled successfully."
                return confirmation_msg, {"action": "appointment_confirmation", "slot": selected_slot}
            else:
                return "⚠️ I couldn't find that time slot. Please choose one from the available options.", None

        # ✅ Predict intent from user input
        intents = self.predict_class(user_input)
        response, tag, pricing = self.get_response(intents, self.knowledge_base)

        # Store conversation context
        state["context"] = tag

        # ✅ Handle appointment booking intent
        if tag == "appointment":
            state["followup"] = "appointment_scheduling"
            return self.schedule_appointment(state["legal_area"]), None

        # ✅ Handle pricing information
        if pricing:
            pricing_info = self.generate_pricing_info(tag, pricing)

            # If the chatbot suggests scheduling an appointment, set follow-up state
            if "Would you like to schedule a consultation" in response:
                state["followup"] = "appointment_scheduling"
                state["legal_area"] = tag  # Store the legal area for reference

            return response, pricing_info

        # 🚀 FINAL RETURN TO AVOID NoneType ERROR
        return response, None




def main():
    chatbot = LegalChatbot()
    print("Legal Chatbot: Welcome to Kvinnolag Juristbyrå! How can I assist you today? (type 'quit' to exit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Legal Chatbot: Thank you for contacting Kvinnolag Juristbyrå. Have a great day!")
            break
            
        response, pricing_info = chatbot.handle_conversation(user_input)
        print(f"Legal Chatbot: {response}")
        
        if pricing_info:
            print(f"\nPricing Information: {pricing_info}")

if __name__ == "__main__":
    main()
