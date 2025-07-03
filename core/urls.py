from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'financial'

urlpatterns = [
    path('', views.home, name='home'),
    path('company/<str:edinet_code>/', views.company_detail, name='company_detail'),
    path('api/ai-analysis/<str:edinet_code>/', views.ai_analysis_ajax, name='ai_analysis_ajax'),
    
    # AJAX エンドポイント
    path('api/company/<str:edinet_code>/predictions/', views.get_predictions_ajax, name='predictions_ajax'),
    path('api/company/<str:edinet_code>/clustering/', views.get_clustering_ajax, name='clustering_ajax'),
    path('api/company/<str:edinet_code>/ai-analysis/', views.get_ai_analysis_ajax, name='ai_analysis_ajax'),
    
    # 認証関連URL
    path('accounts/register/', views.register, name='register'),
    path('accounts/login/', auth_views.LoginView.as_view(), name='login'),
    path('accounts/logout/', views.logout_view, name='logout'),
    path('accounts/profile/', views.profile, name='profile'),
    path('accounts/password_change/', auth_views.PasswordChangeView.as_view(success_url='/accounts/profile/'), name='password_change'),
]