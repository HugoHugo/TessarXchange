import pytest
from datetime import datetime


@pytest.mark.asyncio
async def test_time_endpoint():
    response = await pytest.app.client.get("/time")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    current_datetime = datetime.now()
    assert isinstance(response.json().get("time"), datetime)
    assert response.json().get("time") == current_datetime

    @pytest.mark.parametrize(
        "input_string, expected_output",
        [("2023-04-01T10:30:00Z", "2023-04-01T10:30:00Z")],
        names=["input_string", "expected_output"],
    )
    async def test_time_endpoint_iso_format(input_string, expected_output):
        response = await pytest.app.client.get("/time?format=iso")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)
        assert response.json().get("time") == datetime.strptime(
            input_string, "%Y-%m-%dT%H:%M:%S%Z"
        )
from fastapi.testclient import TestClient
import pytest
from main import app


def test_register_user(client: TestClient):
    response = client.post(
        "/register", json={"email": "test@test.com", "password": "testpass"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data

    def test_register_user_invalid(client: TestClient):
        response = client.post("/register", json={"email": "", "password": ""})
        assert response.status_code == 400
from fastapi.testclient import TestClient
import pytest
from main import app, UserIn, UserOut


@pytest.fixture
def client():
    client = TestClient(app)
    return client


def test_create_user(client):
    data = {"email": "testuser@example.com", "password": "password123"}
    response = client.post("/register", json=data)
    assert response.status_code == 200
    assert "access_token" in response.json()

    def test_invalid_email_format(client):
        data = {"email": "invalid-email-format", "password": "password123"}
        response = client.post("/register", json=data)
        assert response.status_code == 422
        assert "Too many invalid inputs" in str(response.content)
from fastapi.testclient import TestClient
import pytest
from main import app


def test_register_user(client: TestClient):
    response = client.post(
        "/register", json={"email": "test@test.com", "password": "testpass"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data

    def test_register_user_invalid(client: TestClient):
        response = client.post("/register", json={"email": "", "password": ""})
        assert response.status_code == 400
from fastapi.testclient import TestClient
import pytest
from main import app, UserIn, UserOut


@pytest.fixture
def client():
    client = TestClient(app)
    return client


def test_create_user(client):
    data = {"email": "testuser@example.com", "password": "password123"}
    response = client.post("/register", json=data)
    assert response.status_code == 200
    assert "access_token" in response.json()

    def test_invalid_email_format(client):
        data = {"email": "invalid-email-format", "password": "password123"}
        response = client.post("/register", json=data)
        assert response.status_code == 422
        assert "Too many invalid inputs" in str(response.content)
from fastapi.testclient import TestClient
import pytest
from main import app


@pytest.fixture
def client():
    with TestClient(app) as SC:
        yield SC

        def test_generate_wallet_endpoint(client):
            response = client.get("/wallet/BTC/1")
            assert response.status_code == 200
            data = response.json()
            assert "Address" in data
from fastapi.testclient import TestClient
import pytest
from main import app, get_wallet_address


@pytest.fixture
def client():
    with TestClient(app) as TC:
        yield TC

        def test_currency_required():
            response = get_wallet_address.as_view()(request=pytest.app.request)
            assert request.path == "/wallet-address/{currency}"
            assert response.status_code == 400
            assert "Currency is required" in str(response.content)

            def test_generate_wallet_address(client):
                response = client.get("/wallet-address/usd")
                data = response.json()
                assert response.status_code == 200
                assert data == "bT1pK4w3z"
