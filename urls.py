from .views import *
from django.urls import path

urlpatterns = [
    path('',index,name='index'),
    path('predict/',predict_url,name='predict_url'),
    path('history/',history_view, name='history_view'),
]
