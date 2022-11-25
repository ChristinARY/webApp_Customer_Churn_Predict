from django.urls import path
from . import views

urlpatterns = [
    #path('', views.index),
path('acceuil/', views.index, name="index"),
    path('clients/', views.clients, name="clients"),
    path('analyses/', views.analyses, name="analyses"),
    path('predictions/', views.predictions, name='predictions'),
    path('resultat/', views.resultat, name="resultat"),
]