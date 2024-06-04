# file: test/test_admin.py

from datetime import timedelta

from fastapi import status

from ..app.routers.auth import create_access_token
from ..app.routers.bert_sentiment import get_current_user, get_db
from ..app.schemas import PredictionInput, TokenData
from .utils import *

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user


def test_healthcheck_sentiment():
    response = client.get("/mlservice/sentiment/healthcheck")
    print(response.json())
    assert response.status_code == status.HTTP_200_OK


class TestMakePredictionsentiment:
    @pytest.fixture(autouse=True)
    def setup(self, test_superuser, test_user_revoked):
        superuser_token_data = TokenData(
            username=test_superuser.username,
            user_id=test_superuser.id,
            role=test_superuser.role,
            has_access_sentiment=test_superuser.has_access_sentiment,
            has_access_emotion=test_superuser.has_access_emotion,
            expires_delta=timedelta(minutes=30),
        )
        self.superuser_token = create_access_token(superuser_token_data)

        revoked_user_token_data = TokenData(
            username=test_user_revoked.username,
            user_id=test_user_revoked.id,
            role=test_user_revoked.role,
            has_access_sentiment=test_user_revoked.has_access_sentiment,
            has_access_emotion=test_user_revoked.has_access_emotion,
            expires_delta=timedelta(minutes=30),
        )
        self.revoked_user_token = create_access_token(revoked_user_token_data)

    def test_endpoint_ok(self):
        input_data = PredictionInput(text="Sample test for prediction")
        headers = {"Authorization": f"Bearer {self.superuser_token}"}
        response = client.post(
            "/mlservice/sentiment/predict", json=input_data.dict(), headers=headers
        )
        assert response.status_code == status.HTTP_200_OK

    def test_not_legit_user_cannot_call_service(self):
        app.dependency_overrides[get_current_user] = override_get_current_user_revoked
        input_data = PredictionInput(text="Sample text for prediction")
        headers = {"Authorization": f"Bearer {self.revoked_user_token}"}
        response = client.post(
            "/mlservice/sentiment/predict", json=input_data.dict(), headers=headers
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_predict_legit_user(self):
        app.dependency_overrides[get_current_user] = override_get_current_user
        input_data = PredictionInput(text="Sample text for prediction")
        headers = {"Authorization": f"Bearer {self.superuser_token}"}
        response = client.post(
            "/mlservice/sentiment/predict", json=input_data.dict(), headers=headers
        )
        assert response.status_code == status.HTTP_200_OK
