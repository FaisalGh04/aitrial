from flask import Flask, render_template, request, jsonify, Response, g  # Import 'g'
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pyttsx3
import time
import os
# Initialize the TTS engine
tts_engine = pyttsx3.init()

app = Flask(__name__)

# Initialize the model and prompt
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="llama2:latest")  # Use the correct model name
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Store conversation context in Flask's g object
@app.before_request
def initialize_context():
    if not hasattr(g, 'context'):
        g.context = ""
    if not hasattr(g, 'MAX_CONTEXT_LENGTH'):
        g.MAX_CONTEXT_LENGTH = 500  # Reduce context size

# Warm up the model
chain.invoke({"context": "", "question": "Hello"})

@app.route("/")
def home():
    return render_template("index2.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    # Handle exit command
    if user_input.lower() == "exit":
        return jsonify({"response": "Goodbye!"})

    # Limit the context size to avoid excessive memory usage
    if len(g.context) > g.MAX_CONTEXT_LENGTH:
        g.context = g.context[-g.MAX_CONTEXT_LENGTH:]

    # Invoke the chain asynchronously
    result = chain.invoke({"context": g.context, "question": user_input})

    # Update the context
    g.context += f"\nUser: {user_input}\nAI: {result}"

    # Speak the response
    tts_engine.say(result)  # Add the response to the TTS queue
    tts_engine.runAndWait()  # Speak the response

    # Return the text response in chunks
    def generate():
        for word in result.split():
            yield f"data: {word}\n\n"
            time.sleep(0.1)  # Simulate a delay between words

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Use Heroku's port or default to 5001
    app.run(debug=False, host="0.0.0.0", port=port)