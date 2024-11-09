from django import forms

class URLForm(forms.Form):
    url = forms.URLField(label='Enter URL', widget=forms.TextInput(attrs={'size': '80'}))