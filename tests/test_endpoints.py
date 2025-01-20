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
from fastapi.testclient import TestClient
import pytest
from main import app, WalletAddress


def test_generate_address_with_valid_data():
    client = TestClient(app)
    # Set up test data
    currency = "BTC"
    user_id = "user123"
    response = client.get("/generate")
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, WalletAddress)
    assert result.currency == currency
    assert result.user_id == user_id
    assert result.address is not None

    def test_generate_address_with_invalid_currency():
        client = TestClient(app)
        # Set up test data with invalid currency
        currency = ""
        user_id = "user123"
        response = client.get("/generate")
        assert response.status_code == 400
        content = response.content
        assert b"Currency and User ID are required." in content

        def test_generate_address_with_invalid_user_id():
            client = TestClient(app)
            # Set up test data with invalid user_id
            currency = "BTC"
            user_id = ""  # Empty string as user id
            response = client.get("/generate")
            assert response.status_code == 400
            content = response.content
            assert b"Currency and User ID are required." in content

            def test_generate_address():
                client = TestClient(app)
                response = client.get("/generate")
                result = response.json()
                assert isinstance(result, WalletAddress)
                wallet_address = WalletAddress(currency="BTC", user_id="user123")
                assert result.currency == wallet_address.currency
                assert result.user_id == wallet_address.user_id
from fastapi.testclient import TestClient
import pytest
from main import app


@pytest.fixture
def client():
    with TestClient(app) as _:
        yield _

        def test_get_wallet(client):
            response = client.get("/wallet/BTC")
            assert response.status_code == 200
            result = response.json()
            assert "currency" in result
            assert "wallet_address" in result

            def test_invalid_currency(client):
                with pytest.raises(HTTPException):
                    response = client.get("/wallet/INVALID_CURRENCY")

                    def test_wallet_initialization_error():
                        wallet = CryptoWallet("BTC")
                        with pytest.raises(HTTPException) as error:
                            wallet.generate_wallet()
                            assert (
                                "Failed to initialize the cryptocurrency wallet."
                                == str(error.value)
                            )
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from main import app, Wallet


def test_get_wallet():
    client = TestClient(app)
    with pytest.raises(HTTPException):
        response = client.get("/wallet?currency=unsupported")
        assert response.status_code == 400
        with pytest.raises(HTTPException):
            response = client.get("/wallet?path=invalid")
            assert response.status_code == 400
            with pytest.raises(HTTPException):
                response = client.get("/wallet")
                wallet_obj = Wallet(currency="Bitcoin", path=0)
                expected_data = {
                    "currency": "Bitcoin",
                    "path": 0,
                    "address": "bc1q8v5gk7f4tjz3c9h2w8",
                }
                assert response.json() == expected_data
                with pytest.raises(HTTPException):
                    response = client.get("/wallet?currency=Bitcoin&path=-1")
                    wallet_obj = Wallet(currency="Bitcoin", path=0)
                    expected_data = {
                        "currency": "Bitcoin",
                        "path": 0,
                        "address": "bc1q8v5gk7f4tjz3c9h2w8",
                    }
                    assert response.json() == expected_data
                    with pytest.raises(HTTPException):
                        response = client.get("/wallet?currency=Bitcoin")
                        wallet_obj = Wallet(currency="Bitcoin", path=0)
                        expected_data = {
                            "currency": "Bitcoin",
                            "path": 0,
                            "address": "bc1q8v5gk7f4tjz3c9h2w8",
                        }
                        assert response.json() == expected_data
                        with pytest.raises(HTTPException):
                            response = client.get("/wallet?path=0")
                            wallet_obj = Wallet(currency="Bitcoin", path=0)
                            expected_data = {
                                "currency": "Bitcoin",
                                "path": 0,
                                "address": "bc1q8v5gk7f4tjz3c9h2w8",
                            }
                            assert response.json() == expected_data
from fastapi.testclient import TestClient
import pytest
from main import app


@pytest.main
def test_generate_wallet_address():
    client = TestClient(app)
    # Test with valid currency and user
    response = client.post("/wallet/Bitcoin/user1")
    assert response.status_code == 200
    data = response.json()
    assert "address" in data
    assert "currency" in data
    # Test with invalid currency
    response = client.post("/wallet/Ethereum/user2")
    assert response.status_code == 400
    assert response.content.decode() == "Unsupported currency"
import pytest
from main import app


@pytest.fixture()
def client():
    with TestClient(app) as TC:
        yield TC

        def test_generate_wallet(client):
            response = client.post(
                "/wallet",
                json={
                    "currency": "Btc",
                    "user_id": 1,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "currency" in data
            assert "user_id" in data
            assert "wallet_address" in data
            private_key = PrivateKey.random()
            wallet_address = Wallet.from_private_key(private_key).address()
            assert data["wallet_address"] == wallet_address
import pytest
from main import app


@pytest.fixture()
def client():
    yield TestClient(app)

    def test_generate_wallet_address(client):
        # Define expected response
        expected_response = {"address": "BTC-123abc"}
        # Send GET request to endpoint
        response = client.get("/generate_wallet_address")
        # Assert status code is 200 (OK)
        assert response.status_code == 200
        # Extract the data from response and compare it with the expected response
        data = response.json()
        assert "address" in data
        assert data["address"] == expected_response["address"]
import os
from fastapi.testclient import TestClient
from main import app, generate_wallet_address


def test_generate_wallet_address():
    client = TestClient(app)
    # Test with supported currency
    response = client.get("/wallet?currency=eth&user=jane")
    result = response.json()
    assert result == {
        "currency": "eth",
        "user": "jane",
        "address": "0x1234567890123456789012345678901234",
    }
    # Test with unsupported currency
    response = client.get("/wallet?currency=btc&user=john")
    assert response.status_code == 400

    def test_generate_wallet_address_private_key():
        os.environ["PRIVATE_KEY"] = "0xb9c5a3f7b1e2d3c4a5"
        client = TestClient(app)
        response = client.get("/wallet?currency=eth&user=jane")
        result = response.json()
        private_key = os.environ["PRIVATE_KEY"]
        address = generate_wallet_address("eth", "jane")
        assert result == {"currency": "eth", "user": "jane", "address": str(address)}
        del os.environ["PRIVATE_KEY"]
import os
import pytest
from fastapi.testclient import TestClient
from main import app, CryptoWallet


def test_valid_currency():
    crypto_wallet = CryptoWallet("BTC")
    assert crypto_wallet.currency == "BTC"

    def test_invalid_currency():
        with pytest.raises(HTTPException):
            crypto_wallet = CryptoWallet("XRP")

            def test_generate_wallet():
                client = TestClient(app)
                response = client.post("/wallet", json={"currency": "BTC"})
                assert response.status_code == 200
                assert response.json() == {"wallet_address": result, "currency": "BTC"}

                @pytest.mark.parametrize("address_length", [32, 64])
                def test_valid_wallet_address(address_length):
                    crypto_wallet = CryptoWallet(currency="BTC")
                    hex_bytes = os.urandom(32)
                    hex_number = hashlib.sha256(hex_bytes).hexdigest()
                    wallet_address = crypto_wallet._convert_to_address(hex_number)
                    assert len(wallet_address) == address_length
from fastapi import HTTPException
import pytest
from main import app, CryptoWallet


@pytest.fixture
def crypto_wallet():
    wallet = CryptoWallet("BTC")
    return wallet


def test_generate_btc_wallet_address(crypto_wallet):
    wallet_address = crypto_wallet.generate_wallet_address()
    assert isinstance(wallet_address, dict)
    assert "wallet_address" in wallet_address
    assert len(wallet_address["wallet_address"]) > 40

    def test_generate_eth_wallet_address(crypto_wallet):
        wallet_address = crypto_wallet.generate_wallet_address()
        assert isinstance(wallet_address, dict)
        assert "wallet_address" in wallet_address
        assert len(wallet_address["wallet_address"]) > 42

        def test_invalid_currency():
            with pytest.raises(HTTPException):
                wallet = CryptoWallet("XXX")
                wallet.generate_wallet_address()
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from main import app


@pytest.fixture()
def client():
    with TestClient(app) as _client:
        yield _client

        def test_generate_wallet_address(client):
            response = client.get("/wallet/BTC/user1")
            assert response.status_code == 200
            wallet_data = response.json()
            assert "wallet_address" in wallet_data

            def test_currency_not_supported_exception(client):
                response = client.get("/wallet/BTS/user2")
                assert response.status_code == 400
                error_data = response.json()
                assert (
                    "detail" in error_data
                    and error_data["detail"] == "Currency not supported"
                )
