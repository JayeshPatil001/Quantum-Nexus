import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.utils import timezone

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
username =None

def generate_info(topic):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"Provide a brief, concise explanation about the topic: {topic}."
        " The explanation should be clear and suitable for beginners. It should be large and comprehensive enough to cover the topic in detail.\n"
)
    response = model.generate_content(prompt)
    return response.text
    


def generate_mcqs(topic, num_questions):
    """Generates multiple-choice questions using Gemini AI."""
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"Generate {num_questions} multiple-choice questions on the topic: {topic}.Each and every question should be completely different.\n"
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

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

@login_required
def getInfo(request,topic):
    """Displays a brief explanation of a topic."""
    info = generate_info(topic)
    return render(request, "info.html", {"info": info})


@login_required
def ask_question(request):
    global username
    username = request.user.username  # Get the logged-in user's username

    """Handles user questions, generates MCQs, and starts a quiz."""
    if request.method == "POST":
        topic = request.POST.get("topic")
        num_questions = int(request.POST.get("num_questions", 5))
        action = request.POST.get("action")  # Get which button was clicked
        if action == "mcq":
            mcq_list = generate_mcqs(topic, num_questions)  # Ensure this returns a list of dicts

            # Store questions and tracking in session
            request.session["quiz_questions"] = mcq_list
            request.session["current_question_index"] = 0
            request.session["user_answers"] = {}

            return redirect("quiz_question")  # Redirect to the first question
        elif action == "info":
            response = generate_info(topic)
            context = {"info": response, "topic": topic}
            return render(request, "info.html", context)
        elif action == "test":
            mcq_list = generate_mcqs(topic, num_questions)  # Ensure this returns a list of dicts

            # Store questions and tracking in session
            request.session["quiz_questions"] = mcq_list
            request.session["current_question_index"] = 0
            request.session["user_answers"] = {}

            return redirect("test_question")  # Redirect to the first question
    context = {"username": username}
    return render(request, "ask_question.html",context)


@login_required
def quiz_question(request):
    """Displays one MCQ at a time and processes user responses."""
    quiz_questions = request.session.get("quiz_questions", [])
    current_index = request.session.get("current_question_index", 0)

    if not quiz_questions or current_index >= len(quiz_questions):
        return redirect("quiz_result")  # Redirect to results after last question

    question = quiz_questions[current_index]

    if request.method == "POST":
        selected_option = request.POST.get("answer")
        request.session["user_answers"][f"q{current_index}"] = selected_option

        # Move to the next question
        request.session["current_question_index"] += 1
        return redirect("quiz_question")

    return render(request, "quiz_question.html", {"question": question, "index": current_index + 1})

@login_required
def test_question(request):
    """Displays one MCQ at a time and processes user responses."""
    quiz_questions = request.session.get("quiz_questions", [])
    current_index = request.session.get("current_question_index", 0)

    if not quiz_questions or current_index >= len(quiz_questions):
        return redirect("test_result")  # Redirect to results after last question

    question = quiz_questions[current_index]

    if request.method == "POST":
        selected_option = request.POST.get("answer")
        request.session["user_answers"][f"q{current_index}"] = selected_option

        # Move to the next question
        request.session["current_question_index"] += 1
        return redirect("test_question")

    return render(request, "test_question.html", {"question": question, "index": current_index + 1})

@login_required
def test_result(request):
    """Calculates, displays, and stores quiz results in the database."""
    quiz_questions = request.session.get("quiz_questions", [])
    user_answers = request.session.get("user_answers", {})
    user = request.user

    score = 0
    results = []

    for index, question in enumerate(quiz_questions):
        correct_answer = question["answer"]
        user_answer = user_answers.get(f"q{index}", "Not Answered")
        is_correct = user_answer == correct_answer

        # Save to database
        QuizAttempt.objects.create(
            user=user,
            question=question["question"],
            option_A=question["options"]["A"],
            option_B=question["options"]["B"],
            option_C=question["options"]["C"],
            option_D=question["options"]["D"],
            correct_answer=correct_answer,
            user_answer=user_answer if user_answer != "Not Answered" else None,
            is_correct=is_correct
        )

        if is_correct:
            score += 1

        results.append({
            "question": question["question"],
            "user_answer": user_answer,
            "correct_answer": correct_answer
        })

    # Clear session after quiz is completed
    request.session.pop("quiz_questions", None)
    request.session.pop("current_question_index", None)
    request.session.pop("user_answers", None)

    return render(request, "quiz_result.html", {
        "score": score,
        "total": len(quiz_questions),
        "results": results
    })


from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import QuizAttempt
from django.contrib.auth.models import User


@login_required
def quiz_result(request):
    """Calculates, displays, and stores quiz results in the database."""
    quiz_questions = request.session.get("quiz_questions", [])
    user_answers = request.session.get("user_answers", {})
    user = request.user

    score = 0
    results = []

    for index, question in enumerate(quiz_questions):
        correct_answer = question["answer"]
        user_answer = user_answers.get(f"q{index}", "Not Answered")
        is_correct = user_answer == correct_answer

        # Save to database
        QuizAttempt.objects.create(
            user=user,
            question=question["question"],
            option_A=question["options"]["A"],
            option_B=question["options"]["B"],
            option_C=question["options"]["C"],
            option_D=question["options"]["D"],
            correct_answer=correct_answer,
            user_answer=user_answer if user_answer != "Not Answered" else None,
            is_correct=is_correct
        )

        if is_correct:
            score += 1

        results.append({
            "question": question["question"],
            "user_answer": user_answer,
            "correct_answer": correct_answer
        })

    # Clear session after quiz is completed
    request.session.pop("quiz_questions", None)
    request.session.pop("current_question_index", None)
    request.session.pop("user_answers", None)

    return render(request, "quiz_result.html", {
        "score": score,
        "total": len(quiz_questions),
        "results": results
    })

def quiz_history(request):
    """Displays the quiz history of the logged-in user."""
    # get the user using username
    if request.user.is_authenticated:
        print(request.user.username)
        print(request.user)  # Directly prints the user object
    else:
        print("User is not logged in")
    quiz_attempts = QuizAttempt.objects.filter(user=request.user)
    print(quiz_attempts)
    return render(request, "quiz_history.html", {"attempts": quiz_attempts})    


def home(request):
    return render(request, 'home.html')

def register(request):
    global username
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def user_login(request):
    global username
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            username = user.username
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('home')

# create a feedback function that would display feedback page and on post request would store feedback
# feedback has fields, user and feedback
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Feedback

@login_required
def feedback(request):
    if request.method == "POST":
        comment = request.POST.get("comment")
        rating = int(request.POST.get("rating", 5))
        Feedback.objects.create(user=request.user, comment=comment, rating=rating)
        return redirect("feedback")  # Reload feedback page after submission

    feedbacks = Feedback.objects.all().order_by("-created_at")  # Fetch all feedback
    return render(request, "feedback.html", {"feedbacks": feedbacks})
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import Certification, AchievedCertification

@login_required
def certifications(request):
    all_certifications = Certification.objects.all()
    achieved = AchievedCertification.objects.filter(user=request.user).values_list('certification_id', flat=True)

    achieved_certs = Certification.objects.filter(id__in=achieved)
    not_achieved_certs = Certification.objects.exclude(id__in=achieved)

    # Calculate the total number of certifications and total credits
    total_certifications = achieved_certs.count()
    total_credits = sum(cert.credits for cert in achieved_certs)

    # Profile info
    user_profile = {
        'username': request.user.username,
        'total_certifications': total_certifications,
        'total_credits': total_credits
    }

    context = {
        'achieved_certs': achieved_certs,
        'not_achieved_certs': not_achieved_certs,
        'user_profile': user_profile  # Add profile info to context
    }
    return render(request, 'certifications.html', context)


from django.shortcuts import get_object_or_404
from .models import Certification

from django.shortcuts import get_object_or_404
from .models import Certification
from django.contrib.auth.decorators import login_required

@login_required
def start_certification_quiz(request, cert_id):
    """Start quiz specific to a certification and store in session."""
    cert = get_object_or_404(Certification, id=cert_id)

    # Fetching the certification title as the topic
    topic = cert.title

    # Number of questions to generate
    num_questions = 20  # Can be adjusted as needed

    # Generate MCQs related to the certification's topic
    quiz_questions = generate_mcqs(topic, num_questions)

    # Storing quiz questions in session
    request.session["cert_quiz_questions"] = quiz_questions
    request.session["cert_current_question_index"] = 0
    request.session["cert_user_answers"] = {}
    request.session["certification_id"] = cert.id
    start_detection(request)
    return redirect("certification_quiz_question")



@login_required
def certification_quiz_question(request):
    """Displays one question for certification at a time."""
    quiz_questions = request.session.get("cert_quiz_questions", [])
    current_index = request.session.get("cert_current_question_index", 0)

    if not quiz_questions or current_index >= len(quiz_questions):
        return redirect("certification_quiz_result")

    question = quiz_questions[current_index]

    if request.method == "POST":
        selected_option = request.POST.get("answer")
        request.session["cert_user_answers"][f"q{current_index}"] = selected_option
        request.session["cert_current_question_index"] += 1
        return redirect("certification_quiz_question")

    return render(request, "quiz_question.html", {
        "question": question,
        "index": current_index + 1
    })

from .models import AchievedCertification, Certification

@login_required
def certification_quiz_result(request):
    """Processes result for certification quiz and saves if passed."""
    quiz_questions = request.session.get("cert_quiz_questions", [])
    user_answers = request.session.get("cert_user_answers", {})
    user = request.user

    score = 0
    results = []

    for index, question in enumerate(quiz_questions):
        correct_answer = question["answer"]
        user_answer = user_answers.get(f"q{index}", "Not Answered")
        is_correct = user_answer == correct_answer

        if is_correct:
            score += 1

        results.append({
            "question": question["question"],
            "user_answer": user_answer,
            "correct_answer": correct_answer
        })

    # Mark as achieved if passed
    cert_id = request.session.get("certification_id")
    if score >= len(quiz_questions) * 0.6 and cert_id and detector_thread.face_absent_count < 3:
        cert = get_object_or_404(Certification, id=cert_id)
        
        # Check if the user has already been awarded this certification
        if not AchievedCertification.objects.filter(user=user, certification=cert).exists():
            AchievedCertification.objects.create(
                user=user,
                certification=cert,
                achieved_on=timezone.now()
            )
            # Optionally, send a success message or notification
            success_message = f"Congratulations! You've successfully earned the {cert.title} certification."
        else:
            success_message = f"You've already earned the {cert.title} certification."

    else:
        success_message = "You didn't pass the quiz. Please try again."

    # Clear certification session data
    for key in ["cert_quiz_questions", "cert_current_question_index", "cert_user_answers", "certification_id"]:
        request.session.pop(key, None)

    return render(request, "quiz_result.html", {
        "score": score,
        "total": len(quiz_questions),
        "results": results,
        "success_message": success_message
    })


from django.shortcuts import render, get_object_or_404
from .models import Certification
from django.contrib.auth.decorators import login_required

@login_required
def certificate_view(request, cert_id):
    cert = get_object_or_404(Certification, id=cert_id)
    user = request.user
    return render(request, 'certificate.html', {'cert': cert, 'user': user, 'now': timezone.now()})

import cv2
import mediapipe as mp
import time
import threading

class FaceDetectionThread(threading.Thread):
    def __init__(self, max_absent=3, absent_threshold=4):
        super().__init__()
        self.max_absent = max_absent
        self.absent_threshold = absent_threshold
        self.running = False
        self.face_absent_count = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        pTime = 0
        face_absent_start = None
        face_present = True

        while self.running:
            success, img = cap.read()
            if not success:
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(imgRGB)

            if results.multi_face_landmarks:
                face_present = True
                face_absent_start = None
                for faceLms in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(img, faceLms,
                                               self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
            else:
                if face_present:
                    face_absent_start = time.time()
                    face_present = False
                elif face_absent_start is not None:
                    elapsed = time.time() - face_absent_start
                    if elapsed >= self.absent_threshold:
                        self.face_absent_count += 1
                        print(f"[!] Face missing for {int(elapsed)}s. Count: {self.face_absent_count}")
                        face_absent_start = None

            if self.face_absent_count >= self.max_absent:
                print("No face detected 3 times. Exiting...")
                break

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS : {int(fps)}', (20, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False

    def stop(self):
        self.running = False


# views.py
from django.http import JsonResponse

detector_thread = None

def start_detection(request):
    global detector_thread
    if detector_thread is None or not detector_thread.running:
        detector_thread = FaceDetectionThread()
        detector_thread.start()
        return JsonResponse({'status': 'started'})
    return JsonResponse({'status': 'already running'})

def stop_detection(request):
    global detector_thread
    if detector_thread and detector_thread.running:
        detector_thread.stop()
        detector_thread.join()
        return JsonResponse({'status': 'stopped'})
    return JsonResponse({'status': 'not running'})
