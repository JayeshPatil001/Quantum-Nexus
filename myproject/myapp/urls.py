from django.urls import path
from . import views

urlpatterns = [
    path('', views.user_login, name='login'),
    path('register/', views.register, name='register'),
    path('home/', views.home, name='home'),
    path('logout/', views.user_logout, name='logout'),
    path("ask/", views.ask_question, name="ask_question"),
    path("quiz/",  views.quiz_question, name="quiz_question"),
    path("test/",  views.test_question, name="test_question"),
    path("certifications/",  views.certifications, name="certifications"),
    path('certification/<int:cert_id>/start/', views.start_certification_quiz, name='start_certification_quiz'),
    path("certification/quiz/question/", views.certification_quiz_question, name="certification_quiz_question"),
    path("certification/quiz/result/", views.certification_quiz_result, name="certification_quiz_result"),
    path("quiz/result/",  views.quiz_result, name="quiz_result"),
    path("test/result/",  views.quiz_result, name="test_result"),
    path("history/", views.quiz_history, name="history"),
    path("feedback/", views.feedback, name="feedback"),
    path('certificate/<int:cert_id>/', views.certificate_view, name='certificate_view'),

]


