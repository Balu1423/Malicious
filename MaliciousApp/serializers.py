from rest_framework import serializers
from .models import *

class AuthSerializer(serializers.ModelSerializer):
    class Meta:
        model=Auth
        fields='__all__'