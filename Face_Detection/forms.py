from django import forms
from .models import *

class ResgistrationForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = [
            'face_id',
            'name',
            'address',
            'job',
            'phone',
            'email',
            'bio',
            'image'

        ]