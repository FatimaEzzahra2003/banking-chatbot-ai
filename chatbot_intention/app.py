from flask import Flask, render_template, request, jsonify
import joblib
import re
import spacy
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Initialisation Flask
app = Flask(__name__)

# Chargement des modèles 
svm_model = joblib.load("models/intent_svm.pkl")
le = joblib.load("models/label_encoder.pkl")
model = SentenceTransformer("models/sentence_model")

# Charger le modèle spaCy
nlp = spacy.load("en_core_web_sm")

# Initialiser OpenAI (Put your secrect key)
client = OpenAI(api_key="")  



# Fonction de nettoyage
def clean_text(text):
    """
    Nettoyage simple :
    - minuscule
    - suppression ponctuation
    - suppression espaces multiples
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\d+", "<NUM>", text)
    return text

def analyze_syntax(text):
    """
    Applique :
    - Tokenisation
    - Suppression des stopwords
    - Lemmatisation
    - Filtrage des caractères spéciaux
    Retourne le texte lemmatisé nettoyé.
    """
    if not isinstance(text, str):
        return ""
    
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(lemmas)

# Fonction pour générer réponse
def generate_response(intent, user_message):
    """
    Génère une réponse textuelle basée sur l'intention prédite par SVM,
    mais tient compte du contenu du message pour rendre la réponse cohérente.
    """
    prompt = f"""
You are a helpful banking assistant.
The detected intent of the user is: "{intent}".
You MUST generate a response that aligns with this intent, even if the user's message could suggest otherwise.
Here is the user message: "{user_message}"

Generate a polite, concise and informative response according to the detected intent.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and polite banking assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def chatbot_response(user_input):

    cleaned_text = clean_text(user_input)
    processed_text = analyze_syntax(cleaned_text)
    
    print(f"[DEBUG] Texte nettoyé: {cleaned_text}")
    print(f"[DEBUG] Texte traité: {processed_text}")
    
    embedding = model.encode([processed_text])

    # Predict intent with SVM
    intent_encoded = svm_model.predict(embedding)
    intent = le.inverse_transform(intent_encoded)[0]

    # Debug: see what SVM predicts
    print(f"[DEBUG] SVM predicted intent: {intent}")

    # Generate response according to the SVM intent
    response_text = generate_response(intent, user_input)

    return intent, response_text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        print("User input:", user_input)
        intent, response = chatbot_response(user_input)
        print("Bot response:", response)
        return jsonify({"intent": intent, "response": response})
    except Exception as e:
        print("Error:", e)
        return jsonify({"intent": "error", "response": "Une erreur est survenue."})

if __name__ == "__main__":
    app.run(debug=True)