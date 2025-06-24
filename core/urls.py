from django.urls import path
from . import views

app_name = 'financial'

urlpatterns = [
    path('', views.home, name='home'),
    path('company/<str:edinet_code>/', views.company_detail, name='company_detail'),
   
]