# file: test/test_admin.py

from fastapi import status
from datetime import timedelta

from .utils import *
from ..app.routers.bert_sentiment import get_db, get_current_user
from ..app.models import User
from ..app.schemas import TokenData



app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user




def test_healthcheck_emotion():
    response = client.get('/mlservice/emotion/healthcheck')
    print(response.json())
    assert response.status_code == status.HTTP_200_OK

        
