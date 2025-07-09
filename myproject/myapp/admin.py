from django.contrib import admin
from .models import QuizAttempt, Feedback, Certification, AchievedCertification
# Register your models here.
admin.site.register(QuizAttempt)
admin.site.register(Feedback)
admin.site.register(Certification)
admin.site.register(AchievedCertification)
