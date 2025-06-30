from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UserProfile


class UserRegistrationForm(UserCreationForm):
    """ユーザー登録フォーム"""
    email = forms.EmailField(required=True, label='メールアドレス')
    first_name = forms.CharField(max_length=30, required=True, label='名前')
    last_name = forms.CharField(max_length=30, required=True, label='姓')

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
        labels = {
            'username': 'ユーザー名',
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
            # プロフィールを自動作成
            UserProfile.objects.create(user=user)
        return user


class UserProfileForm(forms.ModelForm):
    """ユーザープロフィールフォーム"""
    class Meta:
        model = UserProfile
        fields = ('student_id', 'university', 'department', 'graduation_year', 'interests')
        widgets = {
            'interests': forms.Textarea(attrs={'rows': 4}),
        }