from django.db import models
from django.contrib.auth.models import User

class QuizAttempt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.TextField()
    option_A = models.CharField(max_length=255)
    option_B = models.CharField(max_length=255)
    option_C = models.CharField(max_length=255)
    option_D = models.CharField(max_length=255)
    correct_answer = models.CharField(max_length=1)  # A, B, C, or D
    user_answer = models.CharField(max_length=1, blank=True, null=True)
    is_correct = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)


from django.db import models
from django.contrib.auth.models import User

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Links to the logged-in user
    comment = models.TextField()
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])  # 1-5 Rating
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback by {self.user.username} - {self.rating}/5"
    

from django.db import models
from django.contrib.auth.models import User

class Certification(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    credits = models.PositiveIntegerField()

    def __str__(self):
        return self.title

class AchievedCertification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    certification = models.ForeignKey(Certification, on_delete=models.CASCADE)
    achieved_on = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.certification.title}"


