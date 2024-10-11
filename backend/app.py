from flask import Flask, jsonify, request, send_from_directory,send_file
from flask_cors import CORS
import pyttsx3
from transformers import pipeline
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pytesseract
from PIL import Image
import io
from google.cloud import texttospeech
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})




# Set the environment variable to point to your service account JSON file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

# Initialize text-to-speech engine

tts_engine = pyttsx3.init()

# Initialize Hugging face text summarization model


summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


# Simple user data simulation 
user_data = {
    "john_doe": {
        "progress": 75,
        "learning_preferences": ["audio","visual"],
        "difficulty": "reading_comprehension"
    }

}

# Example data for personlaized learning recommendations

learning_data = np.array([[1, 70],[2, 50], [3, 80],[4, 90],[5, 60]])
learning_labels = np.array([0,1,0,0,1]) #0: Easy, 1: Hard
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(learning_data,learning_labels)

#Text-to-speech function


# Summarization function (helps dyslexic users process large texts)
@app.route('/summarize', methods=['POST'])
def summarize_text():
    text = request.json.get('text')
    summary = summarizer(text, max_length=75, min_length=45, do_sample=False,clean_up_tokenization_spaces=False)
    return jsonify(summary)

@app.route('/speak', methods=['POST'])
def speak():
    # Extract the summarized text from the request body
    text = request.json.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Initialize the Text-to-Speech client
        client = texttospeech.TextToSpeechClient()

        # Prepare the input text
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Configure the voice settings
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-F",  # Use a calming female voice (e.g., Wavenet-F)
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        # Configure audio output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3  # Output format as MP3
        )

        # Synthesize speech from text
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Write the audio content to a file (or return it as a response)
        with open('output.mp3', 'wb') as out:
            out.write(response.audio_content)
            print('Audio content written to output.mp3')

        return jsonify({"message": "Text spoken", "audio_file": "output.mp3"})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Error synthesizing speech"}), 500


@app.route('/output.mp3', methods=['GET'])
def serve_audio():
    return send_file("output.mp3", mimetype="audio/mpeg")

# OCR function
@app.route('/ocr', methods=['POST'])
def ocr_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the image file
        img = Image.open(file)
        
        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(img)

        return jsonify({'ocr_text': text})

    except Exception as e:
        print(e)
        return jsonify({'error': 'OCR processing failed'}), 500
    


@app.route('/ocrSpeak', methods=['POST'])
def ocrSpeak():
    # Extract the summarized text from the request body
    text = request.json.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Initialize the Text-to-Speech client
        client = texttospeech.TextToSpeechClient()

        # Prepare the input text
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Configure the voice settings
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-F",  # Use a calming female voice (e.g., Wavenet-F)
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        # Configure audio output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3  # Output format as MP3
        )

        # Synthesize speech from text
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Write the audio content to a file (or return it as a response)
        with open('output2.mp3', 'wb') as out:
            out.write(response.audio_content)
            print('Audio content written to output2.mp3')

        return jsonify({"message": "Text spoken", "audio_file": "output2.mp3"})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "Error synthesizing speech"}), 500

@app.route('/output2.mp3', methods=['GET'])
def serve_audiocr():
    return send_file("output2.mp3", mimetype="audio/mpeg")

# Get user progress
@app.route('/user/<username>',methods=['GET'])
def get_user_profile(username):
    if username in user_data:
        return jsonify(user_data[username])
    else:
        return jsonify({"error": "User not found"}), 404
    


# Initialize sample question and answer
# Example dataset of questions and answers
questions_data = []

# Function to synthesize text to speech
def synthesize_speech(text):
    try:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-F",  # Calming female voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Save the synthesized speech to a file
        with open('output3.mp3', 'wb') as out:
            out.write(response.audio_content)

        return 'output3.mp3'

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

@app.route('/output3.mp3', methods=['GET'])
def serve_audio3():
    return send_file("output3.mp3", mimetype="audio/mpeg")

# Route to submit a new question and answer
@app.route('/add_question', methods=['POST'])
def add_question():
    data = request.json
    question = data.get('question')
    answer = data.get('answer')

    if not question or not answer:
        return jsonify({"error": "Question or answer missing"}), 400

    questions_data.append({'question': question, 'answer': answer})
    
    return jsonify({"message": "Question added successfully"}), 201


# Route to check the answer
@app.route('/check_answer', methods=['POST'])
def check_answer():
    data = request.json
    question_text = data.get('question')
    user_answer = data.get('answer')

    for q in questions_data:
        if q['question'] == question_text:
            correct_answer = q['answer']
            if user_answer.lower() == correct_answer.lower():
                # Play the encouragement audio
                audio_file = synthesize_speech("Great job! That's the correct answer.")
                return jsonify({"correct": True, "message": "Correct answer!", "audio_file": audio_file}), 200
            else:
                return jsonify({"correct": False, "message": "Incorrect answer. Try again!"}), 200

    return jsonify({"error": "Question not found"}), 404


# Route to get the dataset (to dynamically show the list of questions)
@app.route('/get_questions', methods=['GET'])
def get_questions():
    return jsonify(questions_data), 200


#Learning recommendation based on progress
@app.route('/recommend',methods=['POST'])
def recommend_learning():
    user_progress = request.json.get('progress')
    difficulty = knn_model.predict([[1, user_progress]])[0]

    if difficulty == 0:
        recommendation = "The next lesson will be an easy one!"

    else:
        recommendation = "You are ready for a more challenging task! Great Job!"
    
    return jsonify({"recommendation": recommendation})



if __name__ == '__main__':
    app.run(debug=True)



