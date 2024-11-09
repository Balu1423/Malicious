from django.db import models

# Create your models here.

class Auth(models.Model):
    username=models.CharField(max_length=200, unique=True)
    email=models.EmailField(unique=True)
    pass1=models.CharField(max_length=16)
    pass2=models.CharField(max_length=16)
    
    def __str__(self):
        return self.username
    
class PredictionHistory(models.Model):
    url = models.URLField(max_length=500)
    predicted_label = models.CharField(max_length=100)
    prediction_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.url} - {self.predicted_label}"