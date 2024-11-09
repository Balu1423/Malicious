from django.urls import path
from . import views

urlpatterns = [
    # path('main/',views.main,name='main'),
    path('signin/',views.signin,name='signin'),
    path('signup/',views.signup,name='signup'),
    path('signout',views.signout,name='signout'),
    # path('',views.home,name='home'),
    path('',views.index,name='index'),
    path('predict/',views.predict_url,name='predict_url'),
    path('history/',views.history_view, name='history_view'),
]
