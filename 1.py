import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def generate_mcqs(topic, num_questions):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"Generate {num_questions} multiple-choice questions on the topic: {topic}.\n"
        "Provide the response in strict JSON format without markdown or extra text.\n"
        "The JSON should be an array of objects with the following format: \n"
        "[{\"question\": \"<question_text>\", \"options\": {\"A\": \"<option1>\", \"B\": \"<option2>\", \"C\": \"<option3>\", \"D\": \"<option4>\"}, \"answer\": \"Correct Option (A/B/C/D)\"}]"
    )
    response = model.generate_content(prompt)
    
    try:
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()  # Remove Markdown code block if present
        mcqs = json.loads(response_text)
    except json.JSONDecodeError:
        print("Error parsing MCQs. Ensure Gemini API returns JSON formatted data.")
        mcqs = []
    return mcqs

if __name__ == "__main__":
    topic = input("Enter the topic: ")
    num_questions = int(input("Enter the number of questions: "))
    mcq_list = generate_mcqs(topic, num_questions)
    
    print("\nGenerated MCQs:\n")
    for i, mcq in enumerate(mcq_list, 1):
        print(f"Q{i}: {mcq['question']}")
        for key, option in mcq['options'].items():
            print(f"   {key}) {option}")
        print(f"Answer: {mcq['answer']}\n")