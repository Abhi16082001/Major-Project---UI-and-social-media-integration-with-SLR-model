from django.urls import path
from . import views

urlpatterns=[
    path('',views.interface, name='home'),
    path('result', views.recognition, name='result')
]