from ..db import get_db
from alembic import context
from contextlib import suppress
from datetime import date as dt_date, timedelta
from datetime import datetime
from datetime import datetime, time
from datetime import datetime, timedelta
from datetime import datetime, timezone
from datetime import timedelta
from enum import Enum
from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException, status
from fastapi import HTTPException
from fastapi import TestClient
from fastapi.test import TestClient
from fastapi.testclient import TestClient
from fastapi.wsgi import WebSocket
from jdatetime import parse
from main import AMMPair, ammpairs_router
from main import Bridge
from main import CollateralPosition, app
from main import (
    ComplianceAttestation,
    validate_compliance_attestation,
    attest_compliance_attestation,
)
from main import CrossChainState, StateVerificationError, CrossChainStateService
from main import CustodyRotation, app
from main import DIDDocument, AttestationState, AttestationRequest, AttestationResponse
from main import DarkPool, app
from main import Derivative, SettlementManager
from main import Identity, DecentralizedIdentityRecoveryApp
from main import (
    LiquidityPool,
    create_liquidity_pool_endpoint,
    get_liquidity_pool_endpoint,
)
from main import MarketMaker, app
from main import MigrationRequest, router
from main import MultiSignatureWallet, approve, disapprove
from main import Oracle, get_oracles
from main import OracleData, validate_oracle_data
from main import (
    PaymentChannel,
    create_payment_channel,
    get_payment_channel,
    update_payment_channel,
    delete_payment_channel,
)
from main import PositionRiskEndpoint
from main import ReputationOracle
from main import ReputationOracle, app
from main import StateVerifier, app
from main import TransactionBatch, create_transaction_batch
from main import app
from main import app, AMMPair
from main import app, AssetBridgingValidation
from main import app, AtomicSwapRequest
from main import app, AuditLog
from main import app, BOTS
from main import app, BridgingRequest, BridgingResponse
from main import app, Collateral
from main import app, ComplianceAttestation, DecentralizedComplianceAttestationService
from main import app, CrossChainState
from main import app, CryptoWallet
from main import app, CryptocurrencyWallet
from main import app, DIDDocument, DIDsRepository
from main import app, DIDManager
from main import app, DIDsManager
from main import app, DebtPosition
from main import (
    app,
    DecentralizedIdentity,
    IdentityVerificationRequest,
    VerificationResult,
)
from main import app, Deposit
from main import app, DepositWithdrawalBatch
from main import app, FeeOptimizationInput
from main import app, FeeStatement
from main import app, FeeTier, calculate_fee_tier
from main import app, HistoricalData
from main import app, InstitutionalPrimeBrokers
from main import app, InventoryHedgingItem
from main import app, LiquidityData
from main import app, LiquidityOptimization
from main import app, LiquidityPool
from main import app, LiquiditySnapshot
from main import app, MarginTransfer
from main import app, Oracle
from main import app, OracleData
from main import app, Order
from main import app, OrderBook
from main import app, Portfolio
from main import app, Position
from main import app, ProofOfReserves
from main import app, RateLimiter
from main import app, Reputation
from main import app, ReputationOracle
from main import app, RiskManager
from main import app, SidechainValidatorNode
from main import app, StressTestData
from main import app, SystemHealth
from main import app, TaxReportRequest
from main import app, Trade
from main import app, TradingFeeRebateSystem
from main import app, TradingSignal
from main import app, TradingStrategyParams
from main import app, Transaction
from main import app, UserIn, UserOut
from main import app, Wallet
from main import app, WalletAddress
from main import app, WalletSecurity
from main import app, WithdrawalApproval
from main import app, YieldStrategy
from main import app, calculate_optimized_marking_curve
from main import app, calculate_portfolio_value
from main import app, generate_random_value
from main import app, generate_wallet_address
from main import app, get_db_engine
from main import app, get_wallet_address
from main import app, migrate_pool
from main import app, rebalance_pool, rebalance_endpoint
from main import app, start_liquidation_auction
from main import app, stress_test_portfolio
from main import app, verify_identity
from main import app, wallet
from main import bulk_order_import
from main import create_alert
from main import generate_compliance_report, app
from main import generate_wallet_address, BitcoinAddress
from main import get_alerts
from main import is_trading_pair_eligible, delist_trading_pair
from main import netting_groups, NettingGroup
from main import otc_router, QuoteRequest
from main import rebalance_tokens
from models import User, Order
from models.collateral import Collateral
from models.position import Position
from pytest import mark
from pytest import mark, raises
from pytest import raises
from pytestws import WebSocket as WsClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from typing import IO
from typing import Optional
from typing import Optional, List
from unittest import mock
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import patch, MagicMock
from urllib.parse import quote
from uuid import uuid4
from yfinance import download as yf_download
from your_app import app
from your_app_validator_node import ValidatorNode, validator_node_router
from your_main import app, KycDocument, UserKyc, KYCManager
from your_module import CurveInput, optimize_curve
from your_module import LiquidityBridge, LiquidityBridgeManager
from your_module import ValidatorNode, ValidatorNodeService
import asyncio
import csv
import json
import os
import pandas as pd
import pytest
import re
import tempfile
import time
import ujson
import unittest
import unittest.mock as mock
import uuid


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


@pytest.fixture
def client():
    with TestClient(app) as SC:
        yield SC

        def test_generate_wallet_endpoint(client):
            response = client.get("/wallet/BTC/1")
            assert response.status_code == 200
            data = response.json()
            assert "Address" in data


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


def test_generate_address_with_valid_data():
    client = TestClient(app)
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
        currency = ""
        user_id = "user123"
        response = client.get("/generate")
        assert response.status_code == 400
        content = response.content
        assert b"Currency and User ID are required." in content

        def test_generate_address_with_invalid_user_id():
            client = TestClient(app)
            currency = "BTC"
            user_id = ""
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


@pytest.main
def test_generate_wallet_address():
    client = TestClient(app)
    response = client.post("/wallet/Bitcoin/user1")
    assert response.status_code == 200
    data = response.json()
    assert "address" in data
    assert "currency" in data
    response = client.post("/wallet/Ethereum/user2")
    assert response.status_code == 400
    assert response.content.decode() == "Unsupported currency"


@pytest.fixture()
def client():
    with TestClient(app) as TC:
        yield TC

        def test_generate_wallet(client):
            response = client.post("/wallet", json={"currency": "Btc", "user_id": 1})
            assert response.status_code == 200
            data = response.json()
            assert "currency" in data
            assert "user_id" in data
            assert "wallet_address" in data
            private_key = PrivateKey.random()
            wallet_address = Wallet.from_private_key(private_key).address()
            assert data["wallet_address"] == wallet_address


@pytest.fixture()
def client():
    yield TestClient(app)

    def test_generate_wallet_address(client):
        expected_response = {"address": "BTC-123abc"}
        response = client.get("/generate_wallet_address")
        assert response.status_code == 200
        data = response.json()
        assert "address" in data
        assert data["address"] == expected_response["address"]


def test_generate_wallet_address():
    client = TestClient(app)
    response = client.get("/wallet?currency=eth&user=jane")
    result = response.json()
    assert result == {
        "currency": "eth",
        "user": "jane",
        "address": "0x1234567890123456789012345678901234",
    }
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


def test_generate_wallet_address_btc():
    client = TestClient(app)
    wallet = CryptocurrencyWallet("BTC")
    response = client.get("/wallet/btc")
    assert response.status_code == 200
    expected_address = "your_bitcoin_address"
    assert wallet.wallet_address == expected_address

    def test_generate_wallet_address_eth():
        client = TestClient(app)
        wallet = CryptocurrencyWallet("ETH")
        response = client.get("/wallet/eth")
        assert response.status_code == 200
        expected_address = "your_ethereum_address"
        assert wallet.wallet_address == expected_address

        def test_invalid_currency_raises_http_exception():
            client = TestClient(app)
            with pytest.raises(HTTPException):
                wallet = CryptocurrencyWallet("XRP")
                response = client.get("/wallet/xrp")
                assert response.status_code == 400


@pytest.fixture()
def wallet_address():

    def _wallet_address(currency: str, user_id: Optional[int] = None):
        if not user_id:
            user_id = secrets.randbits(64)
            return WalletAddress(currency=currency, user_id=user_id)
        wallet = _wallet_address(currency=currency, user_id=user_id)
        return wallet.generate_wallet_address()

    return _wallet_address


def test_generate_wallet_address(wallet_address):
    response = wallet_address("BTC", None)
    assert isinstance(response, WalletAddress)
    address = response.address
    assert len(address) == 34

    def test_invalid_currency_or_user_id():
        app.dependency_overrides[WalletAddress.generate_wallet_address] = wallet_address
        with pytest.raises(HTTPException):
            response = app.test_client().get("/generate-wallet-address")


@pytest.mark.parametrize(
    "data, expected_status",
    [
        ('{"type":"card_payment","currency":"USD"}', 200),
        ('{"type":"invalid_type","currency":"EUR"}', 400),
    ],
)
def test_payment_callback(data, expected_status):
    client = TestClient(app)
    with pytest.raises(HTTPException) as e:
        response = client.post("/payment_callback", data=data)
        assert response.status_code == expected_status
        raise e.value

        @pytest.mark.parametrize(
            "data, expected_error",
            [
                ('{"type":"card_payment","currency":"USD"}', False),
                (
                    '{"type":"invalid_type","currency":"EUR"}',
                    True,
                    "Invalid data received from the payment provider.",
                ),
            ],
        )
        def test_payment_callback_invalid_response(data, expected_error):
            client = TestClient(app)
            with pytest.raises(HTTPException) as e:
                response = client.post("/payment_callback", data=data)
                assert response.status_code == 500
                raise e.value
                if expected_error:
                    assert (
                        "An error occurred while processing the payment callback."
                        in str(response.content)
                    )


@pytest.fixture
def margin_trading_pair_manager():
    manager = MarginTradingPairManager()
    return manager


@pytest.mark.parametrize(
    "error_status_code, error_message",
    [
        (400, "ID already exists for an existing trading pair."),
        (404, "The requested trading pair was not found."),
    ],
)
def test_get_trading_pair_by_id(
    margin_trading_pair_manager, error_status_code, error_message
):
    with pytest.raises(HTTPException) as exc_info:
        MarginTradingPairManager.get_trading_pair_by_id(uuid.uuid4())
        assert exc_info.value.status_code == error_status_code
        assert exc_info.value.detail == error_message

        def test_create_trading_pair(margin_trading_pair_manager):
            new_pair = TradingPair(
                id=uuid.uuid4(), base_currency="BTC", quote_currency="USD"
            )
            with mock.patch.object(
                MarginTradingPairManager, "trading_pairs", new=new_pair.id
            ):
                created_pair = margin_trading_pair_manager.create_trading_pair(new_pair)
                assert created_pair == new_pair

                def test_create_trading_pair_with_duplicate_id(
                    margin_trading_pair_manager,
                ):
                    duplicate_pair = TradingPair(
                        id=uuid.uuid4(), base_currency="BTC", quote_currency="USD"
                    )
                    with pytest.raises(HTTPException) as exc_info:
                        margin_trading_pair_manager.create_trading_pair(duplicate_pair)
                        assert exc_info.value.status_code == 400
                        assert (
                            exc_info.value.detail
                            == "ID already exists for an existing trading pair."
                        )


def test_add_margin_position():
    client = TestClient(app)
    response = client.post("/add-margin-position", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert response.json() == {"symbol": "AAPL", "position": "LONG"}

    def test_remove_margin_position():
        client = TestClient(app)
        client.post("/add-margin-position", json={"symbol": "AAPL", "position": "LONG"})
        response = client.post("/remove-margin-position", json={"symbol": "AAPL"})
        assert response.status_code == 200
        assert response.json() == {"status": "success"}

        def test_add_collateral_asset():
            client = TestClient(app)
            client.post(
                "/add-margin-position", json={"symbol": "AAPL", "position": "LONG"}
            )
            response = client.post(
                "/add-collateral-asset",
                json={"asset_id": "USD", "collateral_asset": "USDC"},
            )
            assert response.status_code == 200
            assert response.json() == {"asset_id": "USD", "collateral_asset": "USDC"}

            def test_remove_collateral_asset():
                client = TestClient(app)
                client.post(
                    "/add-margin-position", json={"symbol": "AAPL", "position": "LONG"}
                )
                response = client.post(
                    "/add-collateral-asset",
                    json={"asset_id": "USD", "collateral_asset": "USDC"},
                )
                assert response.status_code == 200
                assert response.json() == {"status": "success"}
                response = client.post(
                    "/remove-collateral-asset", json={"asset_id": "USD"}
                )
                assert response.status_code == 200
                assert response.json() == {"status": "success"}


app = FastAPI()


def test_attestation():
    proof_of_reserves = ProofOfReserves(1000, 500)
    with pytest.raises(HTTPException) as e:
        app.post(
            "/attestation",
            json={"proof_of_reserves": proof_of_reseres},
            follow_redirects=True,
        )
        assert e.type == "http_exc"
        assert e.value.status_code == 400

        def test_attestation_success():
            proof_of_reserves = ProofOfReserves(1000, 500)
            attestation_response = app.post(
                "/attestation",
                json={"proof_of_reserves": proof_of_reserves},
                follow_redirects=True,
            )
            assert attestation_response.status_code == 200
            content = attestation_response.json()
            assert "attestation_text" in content

            def test_attestation_with_signer():
                proof_of_reserves = ProofOfReserves(1000, 500)
                signer = "Test Signer"
                attestation_response = app.post(
                    "/attestation",
                    json={"proof_of_reserves": proof_of_reserves, "signer": signer},
                    follow_redirects=True,
                )
                assert attestation_response.status_code == 200
                content = attestation_response.json()
                assert "attestation_text" in content
                attestation_text = content["attestation_text"]
                assert signer in attestation_text


@pytest.mark.anyio
async def test_get_gas_optimization_strategies():
    client = TestClient(app)
    response = await client.get("/optimization-strategies")
    assert response.status_code == 200
    assert len(response.json()) > 0

    @pytest.mark.anyio
    async def test_get_gas_optimization_strategy():
        client = TestClient(app)
        strategy_id = 1
        response = await client.get(f"/optimization-strategy/{strategy_id}")
        assert response.status_code == 200
        assert response.content_type == "application/json"
        assert isinstance(response.json()["strategy"], models.Strategy)

        @pytest.mark.anyio
        async def test_update_gas_optimization_strategy():
            client = TestClient(app)
            strategy_id = 1
            updated_strategy = GasOptimization(
                id=1,
                strategy_name="Smart Heating (Updated)",
                description="Adjusts heating based on occupancy.",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            response = await client.put(
                f"/optimization-strategy/{strategy_id}", json=updated_strategy.dict()
            )
            assert response.status_code == 200
            assert "message" in response.json()
            assert "updated_strategy" in response.json()

            @pytest.mark.anyio
            async def test_delete_gas_optimization_strategy():
                client = TestClient(app)
                strategy_id = 1
                response = await client.delete(f"/optimization-strategy/{strategy_id}")
                assert response.status_code == 200
                assert "message" in response.json()
                assert "strategy_id" in response.json()


@pytest.fixture
def state():
    chain_id = "test_chain"
    state_data = {"key": "value"}
    return CrossChainState(chain_id, state_data)


def test_state_verification_success(state):
    state.verified = True
    assert state.verify_state() == True

    def test_state_verification_failure(state):
        with pytest.raises(StateVerificationError):
            state.verify_state()

            @pytest.fixture
            def service():
                app = FastAPI()
                return CrossChainStateService()

            def test_service_add_state_verification(service, state):
                state.chain_id = "test_chain"
                service.add_state_verification(state)
                assert len(service.all_state_verifications) == 1

                def test_service_all_state_verifications_empty(service):
                    assert len(service.all_state_verifications) == 0


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client

        def test_create_compliance_attestation(client):
            attestation = ComplianceAttestation(
                timestamp="2023-01-01T00:00:00Z",
                organization_id=1,
                attesting_party_id=2,
                attested_status=True,
            )
            response = client.post("/compliance-attestation", json=attestation.dict())
            assert response.status_code == 200

            def test_create_compliance_attestation_invalid_client(client):
                with pytest.raises(HTTPException) as ex:
                    response = client.post("/compliance-attestation")
                    print(response.text)
                    assert "Attestation already processed" in str(ex.value)

                    @pytest.mark.parametrize(
                        "status, expected_status", [(True, 200), (False, 400)]
                    )
                    def test_create_compliance_attestation_valid_and_invalid(
                        status, expected_status
                    ):
                        attestation = ComplianceAttestation(
                            timestamp="2023-01-01T00:00:00Z",
                            organization_id=1,
                            attesting_party_id=2,
                            attested_status=status,
                        )
                        response = client.post(
                            "/compliance-attestation", json=attestation.dict()
                        )
                        assert response.status_code == expected_status

                        def test_create_compliance_attestation_invalid_timestamp(
                            client,
                        ):
                            attestation = ComplianceAttestation(
                                timestamp="2023-01-32T00:00:00Z",
                                organization_id=1,
                                attesting_party_id=2,
                                attested_status=True,
                            )
                            with pytest.raises(HTTPException) as ex:
                                response = client.post(
                                    "/compliance-attestation", json=attestation.dict()
                                )
                                assert "Attestation already processed" in str(ex.value)

                                def test_create_compliance_attestation_invalid_data(
                                    client,
                                ):
                                    attestation = ComplianceAttestation(
                                        timestamp="2023-01-01T00:00:00Z",
                                        organization_id=-1,
                                        attesting_party_id=2,
                                        attested_status=True,
                                    )
                                    with pytest.raises(HTTPException) as ex:
                                        response = client.post(
                                            "/compliance-attestation",
                                            json=attestation.dict(),
                                        )
                                        assert "Attestation already processed" in str(
                                            ex.value
                                        )

                                        def test_create_compliance_attestation_invalid_organization_id(
                                            client,
                                        ):
                                            attestation = ComplianceAttestation(
                                                timestamp="2023-01-01T00:00:00Z",
                                                organization_id=-1,
                                                attesting_party_id=2,
                                                attested_status=True,
                                            )
                                            with pytest.raises(HTTPException) as ex:
                                                response = client.post(
                                                    "/compliance-attestation",
                                                    json=attestation.dict(),
                                                )
                                                assert (
                                                    "Attestation already processed"
                                                    in str(ex.value)
                                                )

                                                def test_create_compliance_attestation_invalid_attesting_party_id(
                                                    client,
                                                ):
                                                    attestation = ComplianceAttestation(
                                                        timestamp="2023-01-01T00:00:00Z",
                                                        organization_id=1,
                                                        attesting_party_id=-1,
                                                        attested_status=True,
                                                    )
                                                    with pytest.raises(
                                                        HTTPException
                                                    ) as ex:
                                                        response = client.post(
                                                            "/compliance-attestation",
                                                            json=attestation.dict(),
                                                        )
                                                        assert (
                                                            "Attestation already processed"
                                                            in str(ex.value)
                                                        )

                                                        @pytest.mark.parametrize(
                                                            "status, expected_status",
                                                            [(True, 200), (False, 400)],
                                                        )
                                                        def test_create_compliance_attestation_valid_and_invalid(
                                                            status, expected_status
                                                        ):
                                                            attestation = ComplianceAttestation(
                                                                timestamp="2023-01-01T00:00:00Z",
                                                                organization_id=1,
                                                                attesting_party_id=2,
                                                                attested_status=status,
                                                            )
                                                            response = client.post(
                                                                "/compliance-attestation",
                                                                json=attestation.dict(),
                                                            )
                                                            assert (
                                                                response.status_code
                                                                == expected_status
                                                            )

                                                            def test_create_compliance_attestation_invalid_timestamp_and_response(
                                                                client,
                                                            ):
                                                                attestation = ComplianceAttestation(
                                                                    timestamp="2023-01-32T00:00:00Z",
                                                                    organization_id=1,
                                                                    attesting_party_id=2,
                                                                    attested_status=True,
                                                                )
                                                                with pytest.raises(
                                                                    HTTPException
                                                                ) as ex:
                                                                    response = client.post(
                                                                        "/compliance-attestation",
                                                                        json=attestation.dict(),
                                                                    )
                                                                    assert (
                                                                        "Attestation already processed"
                                                                        in str(ex.value)
                                                                    )

                                                                    def test_create_compliance_attestation_invalid_data_and_response(
                                                                        client,
                                                                    ):
                                                                        attestation = ComplianceAttestation(
                                                                            timestamp="2023-01-01T00:00:00Z",
                                                                            organization_id=-1,
                                                                            attesting_party_id=2,
                                                                            attested_status=True,
                                                                        )
                                                                        with pytest.raises(
                                                                            HTTPException
                                                                        ) as ex:
                                                                            response = client.post(
                                                                                "/compliance-attestation",
                                                                                json=attestation.dict(),
                                                                            )
                                                                            assert (
                                                                                "Attestation already processed"
                                                                                in str(
                                                                                    ex.value
                                                                                )
                                                                            )

                                                                            def test_create_compliance_attestation_invalid_organization_id_and_response(
                                                                                client,
                                                                            ):
                                                                                attestation = ComplianceAttestation(
                                                                                    timestamp="2023-01-01T00:00:00Z",
                                                                                    organization_id=-1,
                                                                                    attesting_party_id=2,
                                                                                    attested_status=True,
                                                                                )
                                                                                with pytest.raises(
                                                                                    HTTPException
                                                                                ) as ex:
                                                                                    response = client.post(
                                                                                        "/compliance-attestation",
                                                                                        json=attestation.dict(),
                                                                                    )
                                                                                    assert (
                                                                                        "Attestation already processed"
                                                                                        in str(
                                                                                            ex.value
                                                                                        )
                                                                                    )


@pytest.fixture()
def client():
    yield TestClient(app)

    def test_correlation_analysis(client):
        response = client.get("/correlation")
        assert response.status_code == 200
        data = response.json()
        assert "correlation_matrix" in data
        corr_matrix = data["correlation_matrix"]


def test_monitor_smart_contract():
    contract = SmartContract("your_contract_address")
    monitor_smart_contract(contract)
    expected_state = {"field1": "value1", "field2": "value2"}
    assert contract.current_state == expected_state


@pytest.fixture
def client():
    with TestClient(app) as FastAPIClient:
        yield FastAPIClient

        def test_list_bots(client):
            response = client.get("/bots")
            assert response.status_code == 200
            assert "bots" in response.json()
            bots = response.json()["bots"]
            for bot in bots:
                assert isinstance(bot, dict)
                assert "id" in bot
                assert "name" in bot
                assert "last_active_at" in bot

                def test_get_bot(client):
                    response = client.get("/bot/1")
                    assert response.status_code == 200
                    assert "bot" in response.json()
                    bot = response.json()["bot"]
                    assert isinstance(bot, dict)
                    assert "id" in bot
                    assert "name" in bot
                    assert "last_active_at" in bot

                    def test_get_bot_invalid_id(client):
                        with pytest.raises(HTTPException):
                            response = client.get("/bot/999")
                            assert response.status_code == 404


@pytest.fixture
def client():
    with TestClient(app) as FastAPIClient:
        yield FastAPIClient

        def test_market_maker_profitability_endpoint(client):
            response = client.get("/market_makers/profitability")
            assert response.status_code == 200


@pytest.fixture
def reputation_oracle():
    oracle = ReputationOracle("test_oracle")
    yield oracle
    oracle.oracle_id = None
    oracle.reputation_score = 0


@pytest.fixture
def client():
    with TestClient(app) as _app:
        yield _app

        def test_get_algorithm_executions(client):
            response = client.get("/algorithm_executions")
            assert response.status_code == 200
            assert len(response.json()) == 3
            for item in response.json():
                assert isinstance(item.id, int)
                assert item.symbol == "AAPL"
                assert item.order_type == "Market"
                assert isinstance(item.quantity, int)
                assert isinstance(item.price, float)
                assert isinstance(item.timestamp, datetime)

                def test_create_algorithm_execution(client):
                    new_data = {
                        "symbol": "AAPL",
                        "order_type": "Market",
                        "quantity": 1000,
                        "price": 120.5,
                    }
                    response = client.post("/algorithm_executions", json=new_data)
                    assert response.status_code == 200
                    assert isinstance(response.json().id, int)
                    assert response.json().symbol == "AAPL"
                    assert response.json().order_type == "Market"
                    assert isinstance(response.json().quantity, int)
                    assert isinstance(response.json().price, float)
                    assert isinstance(response.json().timestamp, datetime)

                    def test_create_algorithm_execution_invalid_data(client):
                        new_data = {
                            "symbol": 123,
                            "order_type": "",
                            "quantity": -1000,
                            "price": -120.5,
                        }
                        response = client.post("/algorithm_executions", json=new_data)
                        assert response.status_code == 400


@pytest.fixture
def client():
    return TestClient(app)


def test_valid_kyc(client):
    response = client.post(
        "/kyc",
        json={
            "name": "John Doe",
            "identity_documents": [
                {"document_type": "Passport", "image": None},
                {"document_type": "ID Card", "image": None},
            ],
        },
    )
    assert response.status_code == 200
    assert response.json() == {"result": "KYC verification successful"}

    def test_invalid_document(client):
        response = client.post(
            "/kyc",
            json={
                "name": "John Doe",
                "identity_documents": [
                    {"document_type": "Passport", "image": None},
                    {
                        "document_type": "invalid_document",
                        "image": File("path/to/image"),
                    },
                ],
            },
        )
        assert response.status_code == 400
        assert "Invalid document type" in str(response.content)


def test_background_job():
    client = TestClient(app)
    response = client.get("/portfolio")
    assert response.status_code == 200
    json_content = response.json()
    assert "user_id" in json_content and "portfolio_value" in json_content
    response = client.get("/portfolio")
    assert response.status_code == 200
    json_content = response.json()
    assert "user_id" in json_content and "portfolio_value" in json_content


def test_tax_report_endpoint():
    client = TestClient(app)
    response = client.get("/non_existent_path")
    assert response.status_code == 404
    response = client.get(
        "/tax-report", params={"start_date": "2023-01-01", "end_date": "2023-02-01"}
    )
    expected_response = {
        "tax_reports": [{"report_date": "2023-01-01", "tax_amount": 1000}]
    }
    assert response.status_code == 200
    assert response.json() == expected_response


app = FastAPI()


class TradingStrategyParams(BaseModel):
    risk_tolerance: float
    investment_amount: float
    expected_return_percentage: float
    volatility_tolerance_percentage: float

    def test_endpoint():
        client = TestClient(app)
        with pytest.raises(HTTPException):
            response = client.get("/endpoint")
            assert response.status_code == 404

            def test_update_trading_strategy_params():
                params = TradingStrategyParams(
                    risk_tolerance=0.2,
                    investment_amount=100000,
                    expected_return_percentage=0.1,
                    volatility_tolerance_percentage=0.15,
                )
                client = TestClient(app)
                response = client.put("/trading-strategy-params", json=params)
                assert response.status_code == 200

                def test_update_trading_strategy_params_with_incorrect_data():
                    params = TradingStrategyParams(
                        risk_tolerance=-0.2,
                        investment_amount="not a number",
                        expected_return_percentage=0.1,
                        volatility_tolerance_percentage=0.15,
                    )
                    client = TestClient(app)
                    with pytest.raises(HTTPException):
                        response = client.put("/trading-strategy-params", json=params)
                        assert response.status_code == 400


def create_test_client():
    return TestClient(app)


@pytest.mark.parametrize(
    "identity_public_key, identity_address",
    [("public_key_1", "address_1"), ("public_key_2", "address_2")],
)
def test_identity_verification(identity_public_key, identity_address):
    client = create_test_client()
    identity = DecentralizedIdentity(identity_public_key)
    request = IdentityVerificationRequest(
        identity_public_key=identity.public_key, identity_address=identity.address
    )
    response = client.post("/verify", json=request.dict())
    assert response.status_code == 200
    expected_result = VerificationResult(verified=True, timestamp=datetime.now())
    assert response.json() == expected_result.dict()


@pytest.fixture()
def client():
    return TestClient(app)


def test_create_atomic_swap(client):
    request_data = AtomicSwapRequest(
        request_id="swap1",
        sender_address="sender_address_1",
        receiver_address="receiver_address_1",
        amount=100,
        expiry=datetime.now(),
    )
    response = client.post("/atomic-swap", json=request_data.dict())
    assert response.status_code == 200
    result = response.json()
    assert "swap_id" in result and "status" in result and ("transaction_hash" in result)

    def test_get_atomic_swap(client):
        with pytest.raises(HTTPException):
            response = client.get("/atomic-swap/swap1")
            assert response.status_code == 404

            def test_invalid_swap_id(client):
                with pytest.raises(HTTPException):
                    response = client.get("/atomic-swap/invalid_swap_id")
                    assert response.status_code == 404


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client

        def test_create_update_delete_collaterals(test_client):
            response1 = test_client.get("/collaterals/{collateral_id}")
            assert response1.status_code == 404
            new_collateral_data = Collateral(
                id=str(uuid.uuid4()), chain_id="test_chain", asset="BTC", amount=10
            )
            response2 = test_client.post("/collaterals", json=new_collateral_data)
            assert response2.status_code == 200
            assert response2.json() == new_collateral_data.dict()
            response3 = test_client.get(f"/collaterals/{new_collateral_data.id}")
            assert response3.status_code == 200
            assert response3.json() == new_collateral_data.dict()
            updated_collateral_data = Collateral(
                id=new_collateral_data.id,
                chain_id="updated_test_chain",
                asset="ETH",
                amount=15,
            )
            response4 = test_client.put(
                f"/collaterals/{new_collateral_data.id}", json=updated_collateral_data
            )
            assert response4.status_code == 200
            assert response4.json() == updated_collateral_data.dict()
            response5 = test_client.delete(f"/collaterals/{updated_collateral_data.id}")
            assert response5.status_code == 204

            def test_get_update_delete_non_existent_collateral(test_client):
                response1 = test_client.get("/collaterals/{collateral_id}")
                assert response1.status_code == 404
                new_collateral_data = Collateral(
                    id=str(uuid.uuid4()), chain_id="test_chain", asset="BTC", amount=10
                )
                response2 = test_client.post("/collaterals", json=new_collateral_data)
                assert response2.status_code == 200
                assert response2.json() == new_collateral_data.dict()
                updated_collateral_data = Collateral(
                    id=str(uuid.uuid4()),
                    chain_id="updated_test_chain",
                    asset="ETH",
                    amount=15,
                )
                response3 = test_client.put(
                    f"/collaterals/{new_collateral_data.id}",
                    json=updated_collateral_data,
                )
                assert response3.status_code == 404
                response4 = test_client.delete(f"/collaterals/{new_collateral_data.id}")
                assert response4.status_code == 404


def test_get_volatility():
    client = TestClient(app)
    response = client.get("/volatility")
    data = response.json()
    assert response.status_code == 200
    assert "volatility" in data.keys()


def test_create_validator_node():
    client = TestClient(app)
    response = client.post("/sidechain/validator/node", json={"name": "Test Node"})
    assert response.status_code == 200
    data = response.json()
    node: SidechainValidatorNode = data
    assert isinstance(node, SidechainValidatorNode)

    def test_get_validator_node():
        client = TestClient(app)
        response = client.post("/sidechain/validator/node", json={"name": "Test Node"})
        assert response.status_code == 200
        data = response.json()
        node: SidechainValidatorNode = data
        response = client.get(f"/sidechain/validator/node/{node.id}")
        assert response.status_code == 200
        data = response.json()
        retrieved_node: SidechainValidatorNode = data
        assert isinstance(retrieved_node, SidechainValidatorNode)
        assert node.name == retrieved_node.name

        def test_update_validator_node():
            client = TestClient(app)
            response = client.post(
                "/sidechain/validator/node", json={"name": "Test Node"}
            )
            assert response.status_code == 200
            data = response.json()
            node: SidechainValidatorNode = data
            new_name = "Updated Test Node"
            response = client.put(
                f"/sidechain/validator/node/{node.id}", json={"name": new_name}
            )
            assert response.status_code == 200
            data = response.json()
            updated_node: SidechainValidatorNode = data
            assert isinstance(updated_node, SidechainValidatorNode)
            assert node.name != updated_node.name

            def test_delete_validator_node():
                client = TestClient(app)
                response = client.post(
                    "/sidechain/validator/node", json={"name": "Test Node"}
                )
                assert response.status_code == 200
                data = response.json()
                node: SidechainValidatorNode = data
                response = client.delete(f"/sidechain/validator/node/{node.id}")
                assert response.status_code == 200
                data = response.json()
                deleted_node: SidechainValidatorNode = data
                assert isinstance(deleted_node, SidechainValidatorNode)
                assert node.id != deleted_node.id
                with pytest.raises(HTTPException):
                    client.get(f"/sidechain/validator/node/{deleted_node.id}")


@pytest.fixture
def client():
    with TestClient(app) as ac:
        yield ac

        def test_liquidate_auction_endpoint(client):
            response = client.get("/liquidate")
            assert response.status_code == 200
            data = response.json()
            assert "result" in data


def test_zero_kproof_system():
    zero_kproof_system = ZeroKproofSystem()
    with pytest.raises(HTTPException) as e:
        challenge = "TestChallenge"
        response = zero_kproof_system.generate_proof(challenge=challenge)
        assert str(e.value) == "Invalid challenge"
        proof = zero_kproof_system.generated_proof
        assert isinstance(proof, str)

        def test_generate_proof():
            zero_kproof_system = ZeroKproofSystem()
            with pytest.raises(HTTPException) as e:
                response = zero_kproof_system.generate_proof(challenge=None)
                assert str(e.value) == "Invalid challenge"
                challenge = "TestChallenge"
                response = zero_kproof_system.generate_proof(challenge=challenge)
                assert isinstance(response, dict)

                def test_verify_proof():
                    zero_kproof_system = ZeroKproofSystem()
                    with pytest.raises(HTTPException) as e:
                        proof = "invalid_proof"
                        response = zero_kproof_system.verify_proof(proof=proof)
                        assert str(e.value) == "Invalid proof"
                        challenge = "TestChallenge"
                        encoded_proof = zero_kproof_system.generate_proof(
                            challenge=challenge
                        )
                        response = zero_kproof_system.verify_proof(proof=encoded_proof)
                        assert isinstance(response, bool)

                        def test_endpoint(client: TestClient):
                            response = client.get("/generate-proof")
                            assert response.status_code == 400
                            assert b"Invalid challenge" in response.content
                            response = client.post(
                                "/generate-proof",
                                content_type="application/json",
                                data=json.dumps({"challenge": "TestChallenge"}),
                            )
                            assert response.status_code == 200
                            assert b"Proof generated successfully" in response.content
                            assert isinstance(response.json(), dict)


@pytest.main
def test_risk_decomposition_endpoint():
    client = TestClient(app)
    with pytest.raises(Exception):
        response = client.get("/risk-decomposition")
        assert response.status_code == 404


@pytest.main
def test_optimize_vault_strategy():
    client = TestClient(app)
    response = client.post("/optimize_strategy")
    assert response.status_code == 200
    data = response.json()
    strategy_id = data["strategy_id"]
    num_assets = data["num_assets"]
    asset_allocation = data["asset_allocation"]
    assert len(strategy_id) == 8
    assert num_assets >= 5 and num_assets <= 10
    total = sum(asset_allocation)
    assert total == 1


@pytest.fixture()
def collateral_position():
    position = CollateralPosition(
        id=uuid.uuid4(),
        protocol="example_protocol",
        token_address="0xExampleTokenAddress",
        amount=100.0,
        collateral_factor=1.2,
        last_updated=datetime.now(),
        collateral_positions={},
        router=CollateralPosition.router,
    )
    return position


def test_create_collateral_position(collateral_position):
    response = collateral_position.create_collateral_position(
        position=collateral_position
    )
    assert response.status_code == 200
    assert response.json() == {
        "message": "Collateral position created successfully.",
        "position_id": collateral_position.id,
    }

    def test_get_collateral_position():
        position = CollateralPosition(
            id=uuid.uuid4(),
            protocol="example_protocol",
            token_address="0xExampleTokenAddress",
            amount=100.0,
            collateral_factor=1.2,
            last_updated=datetime.now(),
            collateral_positions={},
            router=CollateralPosition.router,
        )
        position.create_collateral_position(position=position)
        response = position.get_collateral_position(id=position.id)
        assert response.status_code == 200
        assert response.json() == {
            "message": "Collateral position retrieved successfully.",
            "position": position,
        }


def test_load_portfolio_data():
    portfolio = load_portfolio_data()
    assert isinstance(portfolio, Portfolio)
    assert "symbol_map" in dir(portfolio)

    def test_scenario_analysis():
        client = TestClient(app)
        response = client.get("/scenario/AAPL/10.00")
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "scenario_return" in data
        assert "portfolio_value" in data

        def test_scenario_analysis_invalid_symbol():
            client = TestClient(app)
            with pytest.raises(HTTPException):
                response = client.get("/scenario/XOM/20.00")
                assert response.status_code == 404


@pytest.fixture()
def client():
    yield TestClient(app)

    def test_current_balance(client):
        response = client.get("/current-balance")
        assert response.status_code == 200
        result_data = response.json()
        assert isinstance(result_data, dict)
        assert "cryptocurrencies" in result_data
        for data in result_data["cryptocurrencies"]:
            assert "name" in data
            assert "balance" in data

            def test_wallet_creation():
                try:
                    wallet: BitcoinWallet = BitcoinWallet()
                    assert (
                        wallet.cryptocurrencies
                    ), "Expected a list of cryptocurrencies"
                except Exception as e:
                    raise AssertionError(
                        f"An error occurred while trying to create the wallet: {e}"
                    )


importpytest


@pytest.main
def test_calculate_fee_tier():
    volumes = [10000]
    tiered_rates = [0.002, 0.0015]
    expected_output = FeeTier(volume=10000, rate=0.0015)
    assert calculate_fee_tier(volumes) == expected_output

    @pytest.main
    def test_trading_fees(client):
        response = client.get("/trading-fees")
        result = response.json()
        fee_tier = FeeTier(volume=10000, rate=result["rate"])
        assert response.status_code == 200
        assert isinstance(fee_tier, FeeTier)
        assert fee_tier.volume == 10000
        assert fee_tier.rate == 0.0015


@pytest.mark.parametrize(
    "params,expected_status_code",
    [
        (
            TradingStrategyParams(
                entry_price=100,
                stop_loss_percentage=10.0,
                take_profit_percentage=20.0,
                risk_per_trade=200,
            ),
            200,
        ),
        (
            TradingStrategyParams(
                entry_price=-1,
                stop_loss_percentage=5.0,
                take_profit_percentage=15.0,
                risk_per_trade=-500,
            ),
            400,
        ),
    ],
)
def test_endpoint(params: TradingStrategyParams, expected_status_code):
    client = TestClient(app)
    response = client.post("/trading_strategy_params", json=params.json())
    assert response.status_code == expected_status_code

    @pytest.mark.parametrize(
        "params,expected_exception",
        [
            (
                TradingStrategyParams(
                    entry_price=100,
                    stop_loss_percentage=-20.0,
                    take_profit_percentage=20.0,
                    risk_per_trade=200,
                ),
                ValueError,
            ),
            (
                TradingStrategyParams(
                    entry_price=-1,
                    stop_loss_percentage=5.0,
                    take_profit_percentage=15.0,
                    risk_per_trade=500,
                ),
                HTTPException,
            ),
        ],
    )
    def test_endpoint_invalid_params(params: TradingStrategyParams, expected_exception):
        client = TestClient(app)
        with pytest.raises(expected_exception):
            response = client.post("/trading_strategy_params", json=params.json())
            assert False


@pytest.fixture()
def test_amm_pair():
    return AMMPair(
        token0="USD",
        token1="BTC",
        liquidity_mined_per_block=100,
        liquidity_maxed_out_per_block=500,
    )


def test_add_amm_pair(client, test_amm_pair):
    response = client.post("/amm-pairs", json=test_amm_pair)
    assert response.status_code == 200
    assert len(response.json()) == 1

    def test_get_amm_pairs(client, test_amm_pair):
        response = client.get("/amm-pairs")
        assert response.status_code == 200
        assert len(response.json()) == 2
        for pair in response.json():
            assert isinstance(pair, AMMPair)


def test_margin_health_websocket():
    client = TestClient(app)

    @pytest.mark.asyncio
    async def test():
        response = await client.get("/ws/margin-health")
        assert b'Event="open"' in response.content
        await asyncio.sleep(1)
        ws = await response.automate()
        await ws.close()
        assert b"Connection established" in response.content
        test()

        def test_margin_health_websocket_response():
            client = TestClient(app)
            response = client.get("/ws/margin-health")
            assert response.status_code == 101
            assert response.headers["Upgrade"] == "websocket"
            assert response.headers["Connection"] == "upgrade"


def test_trade_creation():
    trade = Trade(
        symbol="AAPL",
        account_id=1,
        trade_time=datetime(2022, 4, 5, 10, 0),
        quantity=100,
        price=150.0,
    )
    assert trade.symbol == "AAPL"
    assert trade.account_id == 1
    assert trade.trade_time == datetime(2022, 4, 5, 10, 0)
    assert trade.quantity == 100
    assert trade.price == 150.0

    def test_trade_to_string():
        trade = Trade(
            symbol="AAPL",
            account_id=1,
            trade_time=datetime(2022, 4, 5, 10, 0),
            quantity=100,
            price=150.0,
        )
        assert (
            str(trade)
            == "Trade(symbol='AAPL', account_id=1, trade_time=datetime(2022, 4, 5, 10, 0), quantity=100, price=150.0)"
        )


@pytest.fixture
def client():
    with TestClient(app) as _app:
        yield _app

        def test_calculate_dynamic_fee(client):
            base_rate = 1.0
            rate_increase_per_packet = 2.5
            initial_congestion_level = 0.8
            network_congestion = NetworkCongestion()
            response = client.get("/calculate_dynamic_fee")
            expected_congestion_level = network_congestion.congestion_level + 0.02
            assert response.status_code == 200
            assert response.json() == {
                "dynamic_fees": float(
                    network_congestion.calculate_dynamic_fee(
                        network_congestion, base_rate, rate_increase_per_packet
                    )
                )
            }
            assert (
                round(expected_congestion_level - initial_congestion_level, 2) == 0.02
            )

            def test_calculate_dynamic_fee_invalid_client():
                with pytest.raises(HTTPException):
                    calculate_dynamic_fee(None, 1.0, 2.5)


@pytest.fixture
def client():
    with TestClient(app) as _:
        yield _

        def test_get_dids(client):
            response = client.get("/dids")
            assert response.status_code == 200
            assert isinstance(response.json(), list)
            assert all((isinstance(item, str) for item in response.json()))

            def test_integrate_did_endpoint(client):
                response = client.post(
                    "/dids/mynewdid", data={"identifier": "mynewdid"}
                )
                assert response.status_code == 200
                assert response.json() == "DID integration successful."

                @pytest.mark.parametrize(
                    "input_identifier, status", [("invalid-did", False)]
                )
                def test_integrate_did_invalid_identifier(
                    client, input_identifier, status
                ):
                    response = client.post("/dids/" + input_identifier)
                    assert response.status_code == 400
                    assert "Invalid or unknown decentralized identifier" in str(
                        response.content
                    )


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_add_liquidity(client):
            response = client.post(
                "/liquidity", json={"symbol": "BTC-USD", "amount": 100}
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
            assert "message" in data


@pytest.mark.anyio
async def test_verify_identity_valid():
    identity_hash = "identity_hash"
    assert verify_identity(identity_hash) == True

    @pytest.mark.anyio
    async def test_verify_identity_invalid_signature():
        identity_hash = "invalid_identity_hash"
        with pytest.raises(ValueError):
            assert verify_identity(identity_hash) == False

            @pytest.mark.anyio
            def test_endpoint():
                response = app.test_client().get("/endpoint")
                assert response.status_code == 404


def test_calculate_optimized_marking_curve():
    optimized_curve = calculate_optimized_marking_curve()
    assert isinstance(optimized_curve, pvector)

    @pytest.mark.asyncio
    async def test_get_optimal_marking_curve():
        client = TestClient(app)
        response = await client.get("/optimal_marking_curve")
        assert response.status_code == 200
        assert isinstance(response.json(), pvector)


@pytest.mark.parametrize(
    "input_data,expected_output",
    [
        ({"user_id": "user123"}, {}),
        ({"user_id": "user456"}, {"message": "KYC registration successful"}),
    ],
)
def test_kyc_endpoint(client, input_data, expected_output):
    response = client.post("/kyc", json=input_data)
    assert response.status_code == 200
    assert response.json() == expected_output


@pytest.fixture
def client():
    yield TestClient(app)

    def test_get_contract_details(client):
        contract_address = "0xContract1"
        response = client.get(f"/contracts/{contract_address}")
        assert response.status_code == 200
        result = response.json()
        assert "status" in result and result["status"] == "monitoring"

        def test_update_contract_status(client):
            contract_address = "0xContract1"
            response = client.post(f"/contracts/{contract_address}")
            assert response.status_code == 200
            content = response.json()
            assert "status" in content and content["status"] == "updated"


importpytest


def test_create_claim():
    client = TestClient(app)
    response = client.post(
        "/claims",
        json={
            "claim_id": str(uuid.uuid4()),
            "policy_number": "A1234567",
            "insured_name": "John Doe",
            "incident_date": datetime(2023, 1, 15),
            "amount_claimed": 10000.0,
            "claim_status": "PENDING",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "claim_id": str(response.json()["claim_id"]),
        "policy_number": "A1234567",
        "insured_name": "John Doe",
        "incident_date": datetime(2023, 1, 15),
        "amount_claimed": 10000.0,
        "claim_status": "PENDING",
    }

    def test_update_claim():
        client = TestClient(app)
        claim_id = str(uuid.uuid4())
        updated_claim_data = InsuranceClaim(
            claim_id=claim_id,
            policy_number="A1234567",
            insured_name="Jane Doe",
            incident_date=datetime(2023, 1, 15),
            amount_claimed=5000.0,
            claim_status="PENDING",
        )
        response = client.put(f"/claims/{claim_id}", json=updated_claim_data)
        assert response.status_code == 200
        updated_claim = response.json()
        expected_claim = InsuranceClaim(
            claim_id=claim_id,
            policy_number="A1234567",
            insured_name="Jane Doe",
            incident_date=datetime(2023, 1, 15),
            amount_claimed=5000.0,
            claim_status="PENDING",
        )
        assert updated_claim == expected_claim

        def test_delete_claim():
            client = TestClient(app)
            claim_id = str(uuid.uuid4())
            response = client.post(
                "/claims",
                json={
                    "claim_id": claim_id,
                    "policy_number": "A1234567",
                    "insured_name": "John Doe",
                    "incident_date": datetime(2023, 1, 15),
                    "amount_claimed": 10000.0,
                    "claim_status": "PENDING",
                },
            )
            assert response.status_code == 200
            claim = response.json()
            updated_claim_id = str(uuid.uuid4())
            response = client.delete(f"/claims/{updated_claim_id}")
            assert response.status_code == 200
            assert response.text == "Claim not found"

            def test_get_claim():
                client = TestClient(app)
                claim_id = str(uuid.uuid4())
                response = client.post(
                    "/claims",
                    json={
                        "claim_id": claim_id,
                        "policy_number": "A1234567",
                        "insured_name": "John Doe",
                        "incident_date": datetime(2023, 1, 15),
                        "amount_claimed": 10000.0,
                        "claim_status": "PENDING",
                    },
                )
                assert response.status_code == 200
                claim = response.json()
                updated_claim_id = str(uuid.uuid4())
                response = client.get(f"/claims/{updated_claim_id}")
                assert response.status_code == 200
                assert response.json() == {
                    "claim_id": updated_claim_id,
                    "policy_number": "A1234567",
                    "insured_name": "John Doe",
                    "incident_date": datetime(2023, 1, 15),
                    "amount_claimed": 10000.0,
                    "claim_status": "PENDING",
                }


def test_endpoint():
    client = TestClient(app)
    response = client.get("/yields")
    assert response.status_code == 200
    data = response.json()
    strategy = data["current_strategy"]
    expected_strategies = [
        "Buy low sell high",
        "Dollar cost average",
        "Rebalancing portfolio",
    ]
    assert strategy in expected_strategies

    def test_yields():
        client = TestClient(app)
        response = client.get("/yields")
        data = response.json()
        assert data["current_strategy"] == "Buy low sell high"


def test_create_account():
    client = TestClient(app)
    account_data = BrokerageAccount(status="inactive")
    response = client.post("/accounts/", json=account_data)
    assert response.status_code == 400
    assert response.json() == {"detail": "Account is inactive."}
    account_data.status = "active"
    response = client.post("/accounts/", json=account_data)
    assert response.status_code == 200
    assert response.json() == {
        "id": str(account_data.id),
        "account_number": account_data.account_number,
        "client_name": account_data.client_name,
        "currency": account_data.currency,
        "status": account_data.status,
    }


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_endpoint(client):
            response = client.get("/delta_hedge")
            assert response.status_code == 200

            def test_invalid_json_payload(client):
                response = client.post("/delta_hedge", data={"invalid": "key"})
                assert response.status_code == 400
                assert (
                    b"Invalid JSON payload received for delta hedging."
                    in response.content
                )


def test_generate_wallet_address():
    with pytest.raises(HTTPException):
        generate_wallet_address(currency="unsupported", user_id=1)
        response = generate_wallet_address(currency="bitcoin", user_id=1)
        assert isinstance(response, dict)
        assert "currency" in response
        assert "user_id" in response
        assert "wallet_address" in response

        def test_bitcoin_address_creation():
            wallet = BitcoinAddress()
            address = wallet.create_address(123)
            if not re.match("^[13][a-km-uptk-z1-9]{27,34}$", str(address)):
                raise AssertionError(
                    "The generated BitcoinAddress is not in the correct format."
                )


def test_batch_orders():
    client = TestClient(app)
    trading_pairs = ["BTC-USDT", "ETH-USD"]
    quantity = 0.01
    price = 50000
    response = client.post(
        "/orders/batch",
        json={"trading_pairs": trading_pairs, "quantity": quantity, "price": price},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["trading_pairs"]) == 2
    assert data["status"] == "success"
    assert "batch_id" in data.keys()


def test_create_margin_trading_position():
    client = TestClient(app)
    response = client.post("/positions", json={})
    assert response.status_code == 200

    def test_get_margin_trading_position():
        client = TestClient(app)
        response = client.get("/positions/1")
        assert response.status_code == 200

        def test_update_margin_trading_position():
            client = TestClient(app)
            response = client.put(
                "/positions/1", json={"symbol": "BTC", "updated_data": {}}
            )
            assert response.status_code == 200

            def test_delete_margin_trading_position():
                client = TestClient(app)
                response = client.delete("/positions/1")
                assert response.status_code == 200

                def test_create_liquidation_thresholds():
                    client = TestClient(app)
                    response = client.post(
                        "/liquidation_thresholds",
                        json={"threshold": LiquidationThreshold()},
                    )
                    assert response.status_code == 200

                    def test_get_liquidation_thresholds():
                        client = TestClient(app)
                        response = client.get("/liquidation_thresholds/1")
                        assert response.status_code == 200

                        def test_update_liquidation_thresholds():
                            client = TestClient(app)
                            response = client.put(
                                "/liquidation_thresholds/1",
                                json={"symbol": "BTC", "updated_data": {}},
                            )
                            assert response.status_code == 200

                            def test_delete_liquidation_thresholds():
                                client = TestClient(app)
                                response = client.delete("/liquidation_thresholds/1")
                                assert response.status_code == 200

                                def test_calculate_position_liquidation_thresholds():
                                    client = TestClient(app)
                                    response = client.get(
                                        "/position_liquidation/BTC",
                                        params={"current_price": 50000, "leverage": 5},
                                    )
                                    assert response.status_code == 200


@pytest.fixture()
def client():
    with TestClient(app) as tc:
        yield tc

        def test_get_liquidity_snapshot(client):
            response = client.get("/liquidity-snapshot")
            assert response.status_code == 200
            liquidity_snapshot: LiquiditySnapshot = response.json()
            assert isinstance(liquidity_snapshot, LiquiditySnapshot)
            assert isinstance(liquidity_snapshot.block_timestamp, datetime)

            def test_get_liquidity_snapshot_invalid_date(client):
                client.get("/liquidity-snapshot?older_than=2")
                response = client.current_response
                assert response.status_code == 400
                snapshot: LiquiditySnapshot = response.json()
                assert isinstance(snapshot, LiquiditySnapshot)
                assert "detail" in snapshot


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client

        def test_bulk_user_permission_update_endpoint(client):
            response = client.get("/bulk-user-permission-update")
            assert response.status_code == 200
            assert (
                "user_id" in response.json()
                and "action" in response.json()
                and ("resource" in response.json())
            )


@pytest.mark.anyio
async def test_create_lending_order_book():
    response = await app.test_client.post("/order_books")
    assert response.status_code == 200
    order_book_data = response.json()
    assert "order_book" in order_book_data

    def test_get_lending_order_book():
        with pytest.raises(HTTPException):
            response = app.test_client.get("/order_books/fake_order_id")
            assert response.status_code == 404
            error_response = response.json()
            assert "detail" in error_response

            def test_update_lending_order_book():
                with pytest.raises(HTTPException):
                    response = app.test_client.put(
                        "/order_books/fake_order_id",
                        json={"new_data": "This is new data"},
                    )
                    assert response.status_code == 404
                    error_response = response.json()
                    assert "detail" in error_response


@pytest.fixture()
def client():
    with TestClient(app) as _client:
        yield _

        def test_stress_test_endpoint(client):
            data = PortfolioData(
                equity=100,
                cash=50,
                risk_free_rate=0.02,
                num_assets=5,
                weights=[0.2, 0.1, 0.3, 0.3, 0.1],
                returns=[0.12, 0.15, 0.08, 0.1, 0.16],
            )

        response = client.post("/stress_test", json=data)
        assert response.status_code == 200


@pytest.mark.parametrize(
    "sender_amount, receiver_amount, expected_status",
    [
        (1e-06, 1e-06, 200),
        (9999999, 9999999, 200),
        (-1, 10000000, 400),
        (1000000.0, -1, 400),
    ],
)
def test_atomic_swap(sender_amount, receiver_amount, expected_status):
    with TestClient(app) as client:
        response = client.get(
            f"/atomic_swap?sender_amount={sender_amount}&receiver_amount={receiver_amount}"
        )
        assert response.status_code == expected_status

        @pytest.mark.parametrize(
            "sender_amount, receiver_amount, exception_message",
            [
                (1e-07, 1e-07, None),
                (9999999999, 9999999999, None),
                (-1, 10000000, "Sender amount must be between 0.000001 and 99999999."),
                (
                    1000000.0,
                    -1,
                    "Receiver amount must be between 0.000001 and 99999999.",
                ),
            ],
        )
        def test_atomic_swap_invalid_data(
            sender_amount, receiver_amount, exception_message
        ):
            with pytest.raises(HTTPException) as ex_info:
                with TestClient(app) as client:
                    response = client.get(
                        f"/atomic_swap?sender_amount={sender_amount}&receiver_amount={receiver_amount}"
                    )
                    assert response.status_code == 400
                    assert ex_info.value.detail == exception_message


@pytest.fixture()
def manager():
    yield DIDsManager()

    def test_create_did(manager):
        identifier = "test-did"
        with pytest.raises(HTTPException) as exc:
            response = manager.create_did(identifier=identifier)
            assert response.status_code == 400
            assert str(exc.value) == "Identifier already exists."
            response = manager.create_did(identifier=identifier)
            did = response["did"]
            assert response["result"] == "DID created successfully."
            assert type(did) is str

            def test_read_did(manager):
                identifier = "test-did"
                with pytest.raises(HTTPException) as exc:
                    response = manager.read_did(identifier=identifier)
                    assert response.status_code == 404
                    assert str(exc.value) == "Identifier not found."
                    response = manager.read_did(identifier=identifier)
                    did = response["did"]
                    assert response["result"] == "DID retrieved successfully."
                    assert type(did) is str

                    def test_update_did(manager):
                        identifier = "test-did"
                        with pytest.raises(HTTPException) as exc:
                            response = manager.update_did(identifier=identifier)
                            assert response.status_code == 404
                            assert str(exc.value) == "Identifier not found."
                            response = manager.create_did(identifier=identifier)
                            did = response["did"]
                            with pytest.raises(HTTPException) as exc:
                                updated_did = stratum.update_DID(
                                    did, new_data="new data"
                                )
                                response = manager.update_did(
                                    identifier=identifier, did=str(updated_did)
                                )
                                assert response.status_code == 404
                                assert str(exc.value) == "Identifier not found."
                                response = manager.update_did(identifier=identifier)
                                updated_did = response["did"]
                                assert response["result"] == "DID updated successfully."
                                assert type(updated_did) is str


app = FastAPI()


class TokenSwapRoute(BaseModel):
    id: uuid.UUID
    from_token: str
    to_token: str
    hops: list[str]
    SWAP_ROUTES = {}

    @pytest.fixture
    def create_route():

        def _create_route(route: TokenSwapRoute):
            if route.id in app.state.SWAP_ROUTES:
                raise HTTPException(status_code=400, detail="Route already exists.")
                app.state.SWAP_ROUTES[route.id] = route
                return route
            return _create_route

        def test_create_token_swap_route(create_route):
            new_route = TokenSwapRoute(
                id=uuid.uuid4(),
                from_token="Token A",
                to_token="Token B",
                hops=["Hop 1", "Hop 2"],
            )
            assert create_route(new_route) == new_route

            def test_list_token_swap_routes():
                create_route = create_route
                route1 = TokenSwapRoute(
                    id=uuid.uuid4(),
                    from_token="Token A",
                    to_token="Token B",
                    hops=["Hop 1", "Hop 2"],
                )
                route2 = TokenSwapRoute(
                    id=uuid.uuid4(),
                    from_token="Token C",
                    to_token="Token D",
                    hops=["Hop 3", "Hop 4"],
                )
                create_route(route1)
                create_route(route2)
                client = TestClient(app)
                response = client.get("/list_token_swap_routes")
                assert response.status_code == 200
                assert len(response.json()["routes"]) == 2


@pytest.fixture
def client():
    yield TestClient(app)

    def test_get_recovery_endpoint(client):
        response = client.get("/recovery")
        assert response.status_code == 200
        assert "This endpoint is for identity recovery." in str(response.content)

        def test_recover_identity_valid_email(client):
            identity_response = {"id": "valid-email-id", "name": "Valid Name"}
            recover_response = {
                "message": "Identity has been successfully recovered. Please check your email for further instructions."
            }
            client.post(
                "/recovery",
                json={
                    "identity_id": "invalid-email-id",
                    "new_email": "test_new_email@example.com",
                },
            )
            with pytest.raises(HTTPException):
                response = client.get("/endpoint")
                response = client.post(
                    "/recovery",
                    json={
                        "identity_id": "valid-email-id",
                        "new_email": "test_new_email@example.com",
                    },
                )
                recovery_response = response.json()
                assert recovery_response["status"] == "SUCCESS"
                assert (
                    recovery_response["message"]
                    == "Identity has been successfully recovered. Please check your email for further instructions."
                )
                assert "valid-email-id" in str(response.content)

                def test_recover_identity_invalid_email(client):
                    with pytest.raises(HTTPException):
                        response = client.post(
                            "/recovery",
                            json={
                                "identity_id": "invalid-email-id",
                                "new_email": "test_new_email@example.com",
                            },
                        )
                        recovery_response = response.json()
                        assert recovery_response["status"] == "ERROR"
                        assert "The email address is already registered." in str(
                            response.content
                        )


def test_optimize_curve():
    curve_input = CurveInput(amount_in=10000, amount_out=11000, fee_rate=0.0003)
    optimized_output = optimize_curve(curve_input)
    assert isinstance(optimized_output, dict)
    assert "x_curve" in optimized_output
    assert "y_curve" in optimized_output

    def test_endpoint():
        client = TestClient(app)
        response = client.get("/optimize_curve")
        assert response.status_code == 200
        assert b'{"x_curve": [...]}' in response.content


@pytest.fixture
def client():
    with TestClient(app) as _client:
        yield _client

        def test_add_oracle_data(client):
            data = OracleData(
                id=uuid.uuid4(), chain="Chain1", timestamp=datetime.now(), value=100.5
            )
            response = client.post("/data", json=data.dict())
            assert response.status_code == 200
            assert "message" in response.json()
            assert "data" in response.json()

            def test_add_oracle_data_bad_request(client):
                data = OracleData(
                    id=uuid.uuid4(),
                    chain="Chain1",
                    timestamp=datetime.now() - datetime(1970, 1, 1),
                    value=100.5,
                )
                response = client.post("/data", json=data.dict())
                assert response.status_code == 400


@pytest.fixture
def client():
    with TestClient(app) as FastAPIClient:
        yield FastAPIClient

        def test_start_liquidation_auction(client):
            response = client.post(
                "/start_liquidation", json={"auction_id": "test_auction"}
            )
            assert response.status_code == 200


def test_fetch_scores():
    security_check = WalletSecurity("https://fakeurl.com")
    latest_scores = security_check.fetch_scores()
    assert isinstance(latest_scores, dict)
    assert "error" not in latest_scores or latest_scores["error"] is None
    assert "scores" in latest_scores

    @pytest.mark.asyncio
    async def test_wallet_security_score():
        response = await app.test_client().post(
            "/wallet-security-score", json={"wallet_url": "https://fakeurl.com"}
        )
        result = response.json()
        assert result["wallet_url"] == "https://fakeurl.com"
        assert isinstance(result["scores"], dict)


def test_unwinding():
    client = TestClient(app)
    symbol = "ABC"
    quantity = 100.0
    price = random.uniform(10.0, 20.0)
    position = Position(symbol=symbol, quantity=quantity, price=price)
    response = client.get(
        "/unwind", params={"symbol": symbol, "quantity": -position.quantity}
    )
    assert response.status_code == 200
    expected_data = {
        "symbol": symbol,
        "price": random.uniform(position.price * 0.9, position.price * 1.1),
        "quantity": -position.quantity,
    }
    assert dict(response.json()) == expected_data

    def test_unwinding_invalid_position():
        client = TestClient(app)
        symbol = "ABC"
        quantity = -100.0
        price = random.uniform(10.0, 20.0)
        position = Position(symbol=symbol, quantity=quantity, price=price)
        with pytest.raises(HTTPException):
            response = client.get(
                "/unwind", params={"symbol": symbol, "quantity": -position.quantity}
            )
            assert response.status_code == 400

            def test_unwinding_invalid_price_range():
                client = TestClient(app)
                symbol = "ABC"
                quantity = 100.0
                price = 1.0
                position = Position(symbol=symbol, quantity=quantity, price=price)
                with pytest.raises(HTTPException):
                    response = client.get(
                        "/unwind",
                        params={"symbol": symbol, "quantity": -position.quantity},
                    )
                    assert response.status_code == 400

                    def test_unwinding_no_position():
                        client = TestClient(app)
                        with pytest.raises(HTTPException):
                            response = client.get("/unwind")
                            assert response.status_code == 404


@pytest.fixture()
def transaction_batch():
    batch_id = "test_batch"
    transactions = [
        {"id": "t1", "sender": "Alice", "receiver": "Bob", "amount": 10},
        {"id": "t2", "sender": "Charlie", "receiver": "David", "amount": 20},
    ]
    timestamp = datetime.now()
    return TransactionBatch(id=batch_id, transactions=transactions, timestamp=timestamp)


def test_create_transaction_batch():
    transaction_batch = transaction_batch()
    with pytest.raises(HTTPException):
        create_transaction_batch({"id": "test_invalid", "transactions": []})
        expected_response = {
            "message": f"Transaction batch {transaction_batch.id} created.",
            "id": transaction_batch.id,
        }
        response = create_transaction_block(transaction_batch.json())
        assert response.status_code == 200
        assert response.json() == expected_response


@pytest.fixture
def test_pool():
    pool_data = LiquidityPoolIn(
        id=str(uuid.uuid4()),
        token1_id="token1",
        token2_id="token2",
        amount1=100,
        amount2=200,
    )
    new_pool = create_liquidity_pool_endpoint(pool_data)
    return new_pool


@pytest.mark.asyncio
async def test_create_liquidity_pool(test_pool):
    assert test_pool.id == str(uuid.uuid4())
    assert test_pool.token1_id == "token1"
    assert test_pool.token2_id == "token2"

    def test_get_liquidity_pool():
        pass


class TestInstitutionalPrimeBrokers:

    def test_valid_institutional_prime_brokers_creation(self):
        id = "123"
        name = "ABC Prime Brokerage"
        address = "123 Main St. New York, NY 10001"
        contact_number = "212-555-0123"
        pb_id = InstitutionalPrimeBrokers(
            id=id, name=name, address=address, contact_number=contact_number
        ).dict()
        assert pb_id["id"] == id
        assert pb_id["name"] == name
        assert pb_id["address"] == address
        assert pb_id["contact_number"] == contact_number

        def test_invalid_institutional_prime_brokers_creation(self):
            with pytest.raises(HTTPException):
                InstitutionalPrimeBrokers(
                    id="123", name="", address="", contact_number=""
                )


def test_place_limit_sell_order():
    symbol = "AAPL"
    price = 150.0
    quantity = 10
    order = Order.place_limit_sell_order(symbol, price, quantity)
    assert isinstance(order, Order)
    assert order.symbol == symbol
    assert order.price == price
    assert order.quantity == quantity

    def test_limit_sell_order_endpoint():
        client = pytest.app.test_client()
        with pytest.raises(HTTPException):
            response = client.get("/limit-sell")

            def test_limit_sell_order_with_symbol_required_error():
                client = pytest.app.test_client()
                response = client.get("/limit-sell", params={"symbol": None})
                assert response.status_code == 400


@pytest.fixture()
def test_client():
    with TestClient(app) as ac:
        yield ac

        def test_endpoint(test_client):
            response = test_client.get("/orderbook?trading_pair=BTC-USDT")
            assert response.status_code == 200
            order_book = response.json()
            assert isinstance(order_book, OrderBook)
            assert "bids" in dir(order_book)
            assert "asks" in dir(order_book)

            def test_empty_order_book(test_client):
                response = test_client.get("/orderbook?trading_pair=INVALID")
                assert response.status_code == 200
                order_book = response.json()
                assert isinstance(order_book, OrderBook)
                assert len(order_book.bids) == 0
                assert len(order_book.asks) == 0


@pytest.fixture
def client():
    with TestClient(app) as _client:
        yield _
        _client.close()

        def test_deposit_timestamp(client):
            deposit_data = Deposit(amount=100)
            response = client.post("/deposits", json=deposit_data.dict())
            assert response.status_code == 200

            def test_deposit_no_timestamp(client):
                deposit_data = Deposit(amount=200, timestamp=None)
                response = client.post("/deposits", json=deposit_data.dict())
                assert response.status_code == 400
                assert "Timestamp cannot be provided" in str(response.content)

                def test_get_deposit_by_id(client):
                    response = client.get("/deposits/1")
                    assert response.status_code == 200
                    assert "deposit_id" in response.json()


importpytest


def test_dependency_overrides():
    client = TestClient(app)
    response = client.get("/dependency_overrides")
    assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_portfolio_background_task():
        client = TestClient(app)
        response = client.post("/update_portfolio_background_task")
        portfolios = await client.get("http://127.0.0.1:8000/portfolio_list")
        for portfolio in portfolios.json():
            assert "updated_at" in portfolio
            assert datetime.now() > datetime.strptime(
                portfolio["updated_at"], "%Y-%m-%d %H:%M:%S"
            )


def test_stop_loss_order():
    order = StopOrder(
        symbol="AAPL", quantity=100, price=150.0, type="stop_loss", trigger_price=145.0
    )
    with pytest.raises(HTTPException):
        with pytest.raises(Exception):
            execute_order(order, market_data)
            assert order.symbol == "AAPL"
            assert order.quantity == 100
            assert order.price == 150.0
            assert order.type == "stop_loss"
            assert order.trigger_price == 145.0

            def test_trailing_stop_loss():
                order = StopOrder(
                    symbol="MSFT",
                    quantity=50,
                    price=175.0,
                    type="trailing_stop",
                    trigger_price=170.0,
                )
                with pytest.raises(HTTPException):
                    with pytest.raises(Exception):
                        market_data = {...}
                        execute_order(order, market_data)
                        assert order.symbol == "MSFT"
                        assert order.quantity == 50
                        assert order.price == 175.0
                        assert order.type == "trailing_stop"
                        assert order.trigger_price == 170.0


def test_update_trading_strategy_params():
    client = TestClient(app)
    params = TradingStrategyParams(
        risk_level=0.1,
        stop_loss_percentage=2.5,
        take_profit_percentage=10.0,
        time_frame_minutes=15,
    )
    response = client.put("/trading-strategy/params", json=params)
    assert response.status_code == 200
    data = response.json()
    expected_data = {
        "status": "parameters updated successfully",
        "updated_params": params.dict(),
    }
    assert data == expected_data


def test_load_snapshot():
    snapshot = LiquiditySnapshot.load_snapshot()
    assert isinstance(snapshot, LiquiditySnapshot)

    def test_save_snapshot():
        timestamp = datetime.datetime.now()
        new_snapshot = LiquiditySnapshot(
            timestamp=timestamp, total_liquidity=0.5, rewards_earned=0.02
        )
        original_snapshot = LiquiditySnapshot.load_snapshot()
        app.save_snapshot(new_snapshot)
        saved_snapshot = LiquiditySnapshot.load_snapshot()
        assert saved_snapshot == new_snapshot
        assert saved_snapshot != original_snapshot

        def test_daily_snapshots():
            client = TestClient(app)
            response = client.get("/liquidity-snapshot")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
            assert "snapshot" in data
            snapshot_data = data["snapshot"]
            assert isinstance(snapshot_data, dict)
            for field_name in ["timestamp", "total_liquidity", "rewards_earned"]:
                assert field_name in snapshot_data


@pytest.mark.parametrize(
    "request_id, expected_exception", [(0, None), (-1, HTTPException)]
)
def test_get_quote(request_id, expected_exception):
    with pytest.raises(expected_exception):
        client = TestClient(app)
        response = client.get(f"/quote/{request_id}")


@pytest.fixture()
def client():
    yield TestClient(app)

    def test_valid_atomic_swap(client):
        atomic_swap_data = {
            "id": str(uuid.uuid4()),
            "from_address": "FromAddress",
            "to_address": "ToAddress",
            "amount": 1.0,
            "expiration_time": datetime.now() + timedelta(minutes=10),
            "secret": "SecretValue",
        }
        response = client.post("/atomic-swap/swap", json=atomic_swap_data)
        assert response.status_code == 200
        assert isinstance(response.json(), dict)
        assert "message" in response.json()

        def test_invalid_atomic_swap(client):
            atomic_swap_data = {
                "id": str(uuid.uuid4()),
                "from_address": "FromAddress",
                "to_address": "ToAddress",
                "amount": 1.0,
                "expiration_time": datetime.now() - timedelta(minutes=10),
                "secret": "SecretValue",
            }
            response = client.post("/atomic-swap/swap", json=atomic_swap_data)
            assert response.status_code == 400
            assert isinstance(response.json(), dict)
            assert "detail" in response.json()


app = FastAPI()


class LendingPosition(BaseModel):
    nft_address: str
    token_id: int
    loan_amount: float
    interest_rate: float
    interest_due_date: datetime

    @classmethod
    def create_lending_position(cls, position: LendingPosition) -> dict:
        return {"position_id": 1234, "status": "active"}

    def test_create_lending_position():
        position = LendingPosition(
            nft_address="0x1", token_id=10, loan_amount=1000.5, interest_rate=0.05
        )
        client = TestClient(app)
        response = client.post("/lending-position", json=position.dict())
        assert response.status_code == 200
        assert "position_id" in response.json()

        def test_create_lending_position_invalid():
            position = LendingPosition(
                nft_address="0x1", token_id=10, loan_amount=-1000.5, interest_rate=-0.05
            )
            client = TestClient(app)
            with pytest.raises(HTTPException):
                client.post("/lending-position", json=position.dict())

                @pytest.mark.parametrize(
                    "loan_amount,interest_rate",
                    [(1000.5, 0.05)],
                    names=["test_create_lending_position_valid"],
                )
                def test_create_lending_position_valid(loan_amount, interest_rate):
                    position = LendingPosition(
                        nft_address="0x1",
                        token_id=10,
                        loan_amount=loan_amount,
                        interest_rate=interest_rate,
                    )
                    client = TestClient(app)
                    response = client.post("/lending-position", json=position.dict())
                    assert response.status_code == 200
                    assert "position_id" in response.json()


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_get_positions(client):
            response = client.get("/positions")
            assert response.status_code == 200
            assert len(response.json()) > 0

            def test_create_position(client):
                new_position = {
                    "id": None,
                    "symbol": "TESTSYMBOL",
                    "quantity": 10.5,
                    "price": 100.2,
                    "timestamp": datetime.now(),
                }
                response = client.post("/positions", json=new_position)
                assert response.status_code == 200
                assert response.json() == new_position

                def test_update_position(client):
                    position_id = 1
                    updated_symbol = "NEW_SYMBOL"
                    updated_price = 150.0
                    response = client.put(
                        f"/positions/{position_id}",
                        json={"symbol": updated_symbol, "price": updated_price},
                    )
                    assert response.status_code == 200
                    assert response.json() == {
                        "id": position_id,
                        "symbol": updated_symbol,
                        "quantity": None,
                        "price": updated_price,
                        "timestamp": datetime.now(),
                    }


@pytest.fixture()
def trading_fee_rebate_system():
    system = TradingFeeRebateSystem()
    yield system
    pass

    def test_add_rebate(trading_fee_rebate_system):
        new_rebate = TradingFeeRebate(
            currency="USD", rebate_percentage=0.05, min_amount=10000
        )
        trading_fee_rebate_system.add_rebate(new_rebate)
        assert len(trading_fee_rebate_system._rebates) == 1
        assert trading_fee_rebate_system._rebates["USD"] == new_rebate

        def test_get_rebate(trading_fee_rebate_system):
            new_rebate = TradingFeeRebate(
                currency="USD", rebate_percentage=0.05, min_amount=10000
            )
            trading_fee_rebate_system.add_rebate(new_rebate)
            response = trading_fee_rebate_system.get_rebate("USD")
            assert len(response) == 1
            assert list(response)[0] == new_rebate


def test_liquidity_routing_endpoint():
    client = TestClient(app)
    response = client.get("/liquidity_routing")
    assert response.status_code == 200
    liquidity_pools = [
        {"id": "pool1", "quantity": 100, "price": 10.5},
        {"id": "pool2", "quantity": 150, "price": 12.8},
    ]

    @pytest.fixture
    def handle_order():
        return app.test_client().post("/order")

    @handle_order.resolves
    def test_handle_order(order):
        assert isinstance(order, Order)


def test_create_oracle():
    oracle = Oracle(name="BitcoinOracle", url="https://bitcoinoracle.com")
    assert len(get_oracles().oracles) == 1
    assert get_oracles().oracles[0].name == "BitcoinOracle"
    assert get_oracles().oracles[0].url == "https://bitcoinoracle.com"

    def test_get_oracle():
        oracle = Oracle(name="BitcoinOracle", url="https://bitcoinoracle.com")
        oracles = [oracle]
        get_oracles.return_value.oracles = oracles
        response = Oracle.get_oracle(oracle.id)
        assert len(response) == 1
        assert response[0].name == "BitcoinOracle"
        assert response[0].url == "https://bitcoinoracle.com"

        def test_get_all_oracles():
            oracle = Oracle(name="BitcoinOracle", url="https://bitcoinoracle.com")
            oracles = [oracle]
            get_oracles.return_value.oracles = oracles
            response = get_oracles().oracles
            assert len(response) == 1
            assert response[0].name == "BitcoinOracle"
            assert response[0].url == "https://bitcoinoracle.com"

            def test_get_invalid_oracle():
                with pytest.raises(HTTPException):
                    Oracle.get_oracle("invalid_oracle_id")


def test_optimize_routing():
    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(app.optimize_routing())
    assert isinstance(data, dict)
    assert "request1" in data
    assert "request2" in data
    assert "request3" in data

    def test_optimize_endpoint():
        client = TestClient(app)
        response = client.get("/optimize")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)
        assert "request1" in response.json()
        assert "request2" in response.json()
        assert "request3" in response.json()

        def test_optimize_endpoint_error():
            client = TestClient(app)
            with pytest.raises(HTTPException):
                response = client.get("/optimize")
                assert response.status_code == 500


def test_create_payment_channel_network():
    client = TestClient(app)
    node1 = {"id": str(uuid.uuid4()), "state": "online"}
    node2 = {"id": str(uuid.uuid4()), "state": "offline"}
    network_input = {"network": PaymentChannelNetwork(nodes=[node1, node2])}
    response = client.post("/networks", json=network_input)
    assert response.status_code == 200
    result_data = response.json()
    assert len(result_data["nodes"]) == 2
    for node in result_data["nodes"]:
        assert "id" in node and "state" in node


@pytest.fixture
def client():
    with TestClient(app) as ac:
        yield ac

        def test_risk_factors_endpoint(client):
            response = client.get("/risk_factors")
            assert response.status_code == 200
            data = response.json()
            risk_factors = data["risk_factors"]
            expected_age_group = "30-40"
            assert risk_factors["age_group"] == expected_age_group
            expected_smoking_status = "Smokes"
            assert risk_factors["smoking_status"] == expected_smoking_status


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_create_reputation(client):
            new_user_id = str(uuid.uuid4())
            reputation = Reputation(new_user_id)
            response = client.get(f"/reputation/{new_user_id}")
            assert response.status_code == 200
            assert response.json() == {"user_id": new_user_id, "reputation": 0}

            def test_update_reputation(client):
                user_id = str(uuid.uuid4())
                reputation = Reputation(user_id)
                response = client.get(f"/reputation/{user_id}")
                assert response.status_code == 200
                data = response.json()
                assert "update_score" in data
                new_score = 50
                response = client.post(
                    f"/reputation/{user_id}/update", json={"new_score": new_score}
                )
                assert response.status_code == 200
                assert response.json() == {"user_id": user_id, "reputation": new_score}

                def test_invalid_update_reputation(client):
                    user_id = str(uuid.uuid4())
                    reputation = Reputation(user_id)
                    response = client.get(f"/reputation/{user_id}")
                    assert response.status_code == 200
                    data = response.json()
                    assert "update_score" in data
                    new_score = -100
                    response = client.post(
                        f"/reputation/{user_id}/update", json={"new_score": new_score}
                    )
                    assert response.status_code == 400
                    assert response.text == "Invalid reputation score"

                    def test_get_user_reputation(client):
                        user_id = str(uuid.uuid4())
                        reputation = Reputation(user_id)
                        response = client.get(f"/reputation/{user_id}")
                        assert response.status_code == 200
                        data = response.json()
                        assert "user_id" in data
                        assert "reputation" in data

                        def test_invalid_user(client):
                            invalid_user_id = "invalid_user"
                            response = client.get(f"/reputation/{invalid_user_id}")
                            assert response.status_code == 404
                            assert response.text == "User not found"


def test_get_inventory_hedging():
    client = TestClient(app)
    response = client.get("/inventory-hedging")
    assert response.status_code == 200
    hedging_items = response.json()
    assert len(hedging_items) > 0

    def test_get_hedging_item_not_found():
        client = TestClient(app)
        symbol = "unknown_symbol"
        with pytest.raises(HTTPException):
            response = client.get(f"/inventory-hedging/{symbol}")
            content = response.content
            assert content
            assert response.status_code == 404


@pytest.mark.parametrize(
    "endpoint, method, expected_status_code",
    [
        ("/collaterals", "POST", 200),
        ("/collaterals", "GET", 200),
        ("/collaterals/{chain}", "PUT", 200),
    ],
)
def test_api_responses(endpoint, method, expected_status_code):
    client = TestClient(app)
    response = getattr(client, method)(f"/endpoint{endpoint}")
    assert response.status_code == expected_status_code


@pytest.mark.anyio
async def test_create_oracle():
    new_oracle = ReputationOracle(id="new_id", domain="test_domain", score=100)
    response = app.test_client().post("/oracle", json=new_oracle.dict())
    assert response.status_code == 200
    assert "new_id" in response.json()

    @pytest.mark.anyio
    async def test_get_oracle():
        oracles = ReputationOracle(id="test_id", domain="test_domain", score=100)
        with pytest.raises(HTTPException) as ex:
            app.test_client().get("/oracle/test_id")
            assert ex.value.status_code == 404

            @pytest.mark.anyio
            async def test_update_oracle():
                oracles = ReputationOracle(
                    id="test_id", domain="test_domain", score=100
                )
                with pytest.raises(HTTPException) as ex:
                    app.test_client().put("/oracle/test_id")
                    assert ex.value.status_code == 404


def test_fee_optimization():
    client = TestClient(app)
    input_data = FeeOptimizationInput(
        token0_price=1.5,
        token1_price=2.3,
        reserve0_amount=1000,
        reserve1_amount=1200,
        fee_cemented=0.001,
        slippage_percentage=0.05,
    )
    response = client.post("/fee_optimization", json=input_data.json())
    assert response.status_code == 200

    def test_fee_optimization_unprocessable_input():
        client = TestClient(app)
        input_data = FeeOptimizationInput(
            token0_price=-1.5,
            token1_price=2.3,
            reserve0_amount=1000,
            reserve1_amount=-1200,
            fee_cemented=0.001,
            slippage_percentage=0.05,
        )
        response = client.post("/fee_optimization", json=input_data.json())
        assert response.status_code == 422


@pytest.fixture
def dark_pool():
    return DarkPool()


def test_create_dark_pool(dark_pool):
    id = str(uuid.uuid4())
    dark_pool.id = id
    dark_pool.buyer = "Buyer A"
    dark_pool.seller = "Seller B"
    dark_pool.buy_amount = 1000.0
    dark_pool.sell_amount = 500.0
    response = dark_pool.dict()
    assert len(response) == 13
    assert response["id"] == id
    assert response["buyer"] == "Buyer A"
    assert response["seller"] == "Seller B"
    assert response["buy_amount"] == 1000.0
    assert response["sell_amount"] == 500.0

    def test_get_dark_pool(dark_pool):
        id = str(uuid.uuid4())
        dark_pool.id = id
        dark_pool.buyer = "Buyer A"
        dark_pool.seller = "Seller B"
        dark_pool.buy_amount = 1000.0
        dark_pool.sell_amount = 500.0
        response = dark_pool.dict()
        get_response = dark_pool.get_dark_pool(id)
        assert len(get_response) == 13
        assert get_response["id"] == id
        assert get_response["buyer"] == "Buyer A"
        assert get_response["seller"] == "Seller B"
        assert get_response["buy_amount"] == 1000.0
        assert get_response["sell_amount"] == 500.0

        def test_get_dark_pool_non_existent(dark_pool):
            id = str(uuid.uuid4())
            with pytest.raises(HTTPException) as context:
                _ = dark_pool.get_dark_pool(id)
                assert str(context.value) == "Dark pool not found."


def test_market_depth():
    response = app.test_client().get("/market_depth")
    assert response.status_code == 200
    data = response.json()
    assert "asset" in data
    assert "buy" in data
    assert "sell" in data


def test_tax_report_endpoint():
    client = TestClient(app)
    start_date_str = "2022-01-01"
    end_date_str = "2022-12-31"
    response = client.get(
        f"/tax-report?start_date={start_date_str}&end_date={end_date_str}"
    )
    assert response.status_code == 200
    response = client.get("/tax-report")
    assert response.status_code == 400
    assert b"Missing required parameters" in response.content
    invalid_start_date_str = "2022-13-01"
    valid_end_date_str = "2022-12-31"
    response = client.get(
        f"/tax-report?start_date={invalid_start_date_str}&end_date={valid_end_date_str}"
    )
    assert response.status_code == 400
    assert b"Invalid start_date or end_date format" in response.content


@pytest.mark.parametrize(
    "alert",
    [
        PriceAlert(
            product_id=1,
            target_price=50.0,
            trigger_time="2023-01-01 10:00",
            notification_type="email",
        ),
        PriceAlert(
            product_id=2,
            target_price=75.0,
            trigger_time="2023-02-01 15:30",
            notification_type="sms",
        ),
    ],
)
def test_create_alert(alert):
    response = create_alert(alert)
    assert response.status_code == 200
    assert response.json() == {"message": "Price alert created successfully."}

    @pytest.mark.parametrize("alert_id", [1, 2])
    def test_get_alerts(alert_id):
        response = get_alerts(alert_id)
        assert response.status_code == 200
        assert response.json() == [
            {
                "product_id": 1,
                "target_price": 50.0,
                "trigger_time": "2023-01-01T10:00:00Z",
                "notification_type": "email",
            },
            {
                "product_id": 2,
                "target_price": 75.0,
                "trigger_time": "2023-02-01T15:30:00Z",
                "notification_type": "sms",
            },
        ]


@pytest.fixture
def client():
    with TestClient(app) as _client:
        yield _client

        def test_system_health(client):
            response = client.get("/system-health")
            assert response.status_code == 200
            health_data: SystemHealth = response.json()
            assert isinstance(health_data, SystemHealth)
            assert "application_uuid" in health_data
            assert "uptime_seconds" in health_data
            assert "memory_usage_mb" in health_data
            assert "cpu_usage_percent" in health_data
            assert isinstance(health_data.application_uuid, str)
            assert isinstance(health_data.uptime_seconds, int)
            assert isinstance(health_data.memory_usage_mb, int)
            assert isinstance(health_data.cpu_usage_percent, float)

            def test_system_health_exception(client):
                response = client.get("/system-health", status_code=500)
                content = response.content
                assert b"Internal Server Error" in content


def test_create_snapshot():
    client = TestClient(app)
    response = client.post(
        "/liquidity_snapshot",
        json={"snapshot_time": "2023-01-01T00:00:00Z", "token_liquidity": 100.0},
    )
    assert response.status_code == 200
    assert b"Snapshot created successfully." in response.content

    def test_get_snapshots():
        client = TestClient(app)
        response = client.get("/liquidity_snapshots")
        assert response.status_class == "200 OK"
        assert len(response.json()) == 2


def test_fee_statement_endpoint():
    client = TestClient(app)
    fee_statement_data = FeeStatement(
        client_id=123,
        statement_date=datetime(2022, 1, 1),
        statement_period="Q4 FY 2021",
        transactions=[
            {"type": "deposit", "amount": 5000},
            {"type": "withdrawal", "amount": 3000},
        ],
    )
    response = client.post("/generate_fee_statement", json=fee_statement_data.dict())
    assert response.status_code == 200
    fees = FeeStatement.parse_raw(response.text)
    assert fees.client_id == fee_statement_data.client_id
    assert fees.statement_date == fee_statement_data.statement_date
    assert fees.statement_period == fee_statement_data.statement_period
    assert len(fees.transactions) == len(fee_statement_data.transactions)

    def test_fee_statement_endpoint_invalid_client_id():
        client = TestClient(app)
        fee_statement_data = FeeStatement(
            client_id=-123,
            statement_date=datetime(2022, 1, 1),
            statement_period="Q4 FY 2021",
            transactions=[
                {"type": "deposit", "amount": 5000},
                {"type": "withdrawal", "amount": 3000},
            ],
        )
        response = client.post(
            "/generate_fee_statement", json=fee_statement_data.dict()
        )
        assert response.status_code == 400
        fees = FeeStatement.parse_raw(response.text)
        assert fees.client_id == fee_statement_data.client_id
        assert fees.statement_date == fee_statement_data.statement_date
        assert fees.statement_period == fee_statement_data.statement_period
        assert len(fees.transactions) == len(fee_statement_data.transactions)


importpytest


def test_get_whitelisted_addresses():
    client = TestClient(app)
    response = client.get("/whitelist")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

    def test_set_whitelisted_address():
        client = TestClient(app)
        response = client.post("/whitelist", json={"address": "0x1234567890"})
        assert response.status_code == 200
        assert response.json() == {"result": "address added to whitelist"}

        def test_get_whitelisted_address():
            client = TestClient(app)
            WHITE_LISTED_ADDRESSES.append("0x1234567890")
            response = client.get("/whitelist/0x1234567890")
            assert response.status_code == 200
            assert response.json() == {
                "result": "Address 0x1234567890 is valid for withdrawal"
            }

            def test_get_whitelisted_address_not_found():
                client = TestClient(app)
                with pytest.raises(HTTPException):
                    response = client.get("/whitelist/0x9999999999")
                    assert response.status_code == 404
                    assert "Address not found in whitelist" in str(response.content)


@pytest.fixture()
def test_data():
    return {"BTC-USD": True, "ETH-BNB": False}


@pytest.mark.parametrize("pair_symbol", ["BTC-USD", "ETH-USD"])
def test_is_trading_pair_eligible(test_data):
    assert is_trading_pair_eligible("BTC-USD") == test_data["BTC-USD"]
    assert is_trading_pair_eligible("ETH-USD") == test_data["ETH-USD"]

    @pytest.mark.parametrize("pair_symbol", ["BTC-USD"])
    def test_delist_trading_pair(test_data):
        expected_output = {"result": "Delisted BTC-USD"}
        response = delist_trading_pair(pair_symbol)
        assert response == expected_output


@pytest.fixture()
def client():
    with TestClient(app) as _client:
        yield _

        def test_stress_test_endpoint(client):
            response = client.get("/stress-test")
            assert response.status_code == 200
            result_data = response.json()
            assert isinstance(result_data, dict)
            assert "portfolio_return" in result_data
            assert "portfolio_volatility" in result_data


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_create_proof_of_reserves(client):
            data = {"stablecoin_total": 1000000, "bank_balance": 1100000}
            response = client.post("/proof_of_reserves", json=data)
            assert response.status_code == 200
            attestation_data = response.json()
            stablecoin_total = attestation_data["attestation_data"]["stablecoin_total"]
            bank_balance = attestation_data["attestation_data"]["bank_balance"]
            assert stablecoin_total == data["stablecoin_total"]
            assert bank_balance == data["bank_balance"]


@pytest.mark.fastapi
class TestAPI:

    @mark.test_staking
    def test_stake(self):
        token_data = Stake(
            stake_id=1, owner="alice", amount=100.0, timestamp=datetime.now()
        )
        with raises(HTTPException, status_code=400, detail="Stake already exists"):
            STAKES[token_data.stake_id] = token_data

            @mark.test_voting
            def test_vote(self):
                vote_data = Vote(
                    vote_id=1,
                    proposal="Proposal 1",
                    voter="alice",
                    votes_for=50,
                    votes_against=10,
                    timestamp=datetime.now(),
                )
                with raises(
                    HTTPException, status_code=400, detail="Vote already exists"
                ):
                    VOTES[vote_data.vote_id] = vote_data

                    @mark.test_stakes_endpoint
                    def test_get_stakes(self):
                        response = self.client.get("/stakes")
                        assert response.status_code == 200
                        data = response.json()
                        assert isinstance(data, dict)
                        assert "STAKES" in data.keys()


@pytest.fixture
def client():
    pass

    def test_add_validator_node(client):
        pass

        def test_get_validator_node(client):
            pass

            def test_update_validator_node(client):
                pass

                def test_delete_validator_node(client):
                    pass


@pytest.fixture
def client():
    fastapi_app = create_fastapi_app()
    yield TestClient(fastapi_app)
    fastapi_app.client.close()

    def test_endpoint(client):
        response = client.get("/manipulate_market")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, MarketManipulation)
        assert data.manipulated_price > 0

        @pytest.mark.asyncio
        async def test_market_manipulation_detection():
            client = TestClient(app)
            response = await client.post(
                "/manipulate_market",
                json={
                    "timestamp": datetime.now(),
                    "manipulated_price": random.uniform(100, 200),
                },
            )
            assert response.status_code == 400

            def test_invalid_timestamp():
                client = TestClient(app)
                response = client.get("/manipulate_market")
                assert response.status_code == 404


@pytest.fixture()
def attestestation_request():
    return AttestationRequest(
        did_document=DIDDocument(id=str(uuid.uuid4())),
        attestation_request_id=str(uuid.uuid4()),
    )


def test_attestation_request_valid(attestestation_request):
    assert not attestestation_request.did_document.id
    assert not attestestation_request.attestation_request_id
    attestestation_request.did_document.id = str(uuid.uuid4())
    attestestation_request.attestation_request_id = str(uuid.uuid4())
    assert attestestation_request.did_document.id
    assert attestestation_request.attestation_request_id

    def test_attestation_request_invalid():
        with pytest.raises(HTTPException):
            AttestationRequest(
                did_document=DIDDocument(id=str(uuid.uuid4())),
                attestation_request_id=str(uuid.uuid4()),
            )
            assert not DIDDocument.id
            assert not uuid.UUID
            request = AttestationRequest(
                did_document=DIDDocument(id=None), attestation_request_id=uuid.uuid4()
            )
            with pytest.raises(HTTPException):
                if not request.did_document.id:
                    raise HTTPException(
                        status_code=400, detail="Missing DID document ID."
                    )


def test_create_event():
    client = TestClient(app)
    event_data = {
        "event_type": str(uuid.uuid4()),
        "timestamp": datetime.now(),
        "recipient_address": "0x12345678901234567890123456789012345678901",
    }
    response = client.post("/event", json=event_data)
    assert response.status_code == 200
    assert event_data["event_type"] in response.json()

    def test_duplicate_event():
        client = TestClient(app)
        event_id = uuid.uuid4()
        event_data = {
            "id": event_id,
            "event_type": "TOKEN_DISTRIBUTION",
            "timestamp": datetime.now(),
            "recipient_address": "0x12345678901234567890123456789012345678901",
        }
        with pytest.raises(HTTPException):
            response = client.post("/event", json=event_data)
            assert response.status_code == 400


app = FastAPI()


class LPRebalancer:

    def __init__(self, lp_address: str):
        self.lp_address = lp_address

        async def rebalance_lps():
            while True:
                lp_rebalancer = LPRebalancer(lp_address="0x...")
                await lp_rebalancer.rebalance()
                print("Waiting for the next rebalance...")
                await asyncio.sleep(60 * 5)

                def test_rebalance_lps():
                    loop = asyncio.get_event_loop()
                    response = loop.run_until_complete(rebalance_lps())
                    assert "Liquidity pool rebalancing initiated." in str(response)

                    @pytest.mark.asyncio
                    async def test_create_endpoint(client: TestClient):
                        response = await client.post("/rebalance_lps")
                        assert response.status_code == 200

                        def test_lp_rebalancer():
                            lp_rebalancer = LPRebalancer("0x...")
                            with pytest.raises(HTTPException):
                                rebalance_result = lp_rebalancer.rebalance()
                                assert "Liquidity pool rebalanced." in str(
                                    rebalance_result
                                )


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_create_collateral(client):
            response = client.post(
                "/collaterals", json={"type": "Equity", "amount": 100000}
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, Collateral)
            assert data.id is not None
            assert len(data.collaterals) == 0

            def test_get_collaterals(client):
                create_collateral(client, type="Equity", amount=100000)
                response = client.get("/collaterals")
                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, list)
                assert all((isinstance(item, Collateral) for item in data))

                def test_create_position(client):
                    create_collateral(client, type="Equity", amount=100000)
                    position_data = {
                        "type": "Long",
                        "collateral_id": "new_collateral",
                        "underlying_asset_id": "new_underlying",
                        "margin_ratio": 0.4,
                        "position_size": 50000,
                    }
                    response = client.post("/positions", json=position_data)
                    assert response.status_code == 200
                    data = response.json()
                    assert isinstance(data, Position)
                    assert data.type is not None
                    assert len(data.collaterals) == 1
                    assert data.collaterals[0].id == position_data["collateral_id"]
                    assert (
                        data.underlying_asset_id == position_data["underlying_asset_id"]
                    )

                    def test_get_positions(client):
                        create_collateral(client, type="Equity", amount=100000)
                        create_position(
                            client,
                            type="Long",
                            collateral_id="new_collateral",
                            underlying_asset_id="new_underlying",
                            margin_ratio=0.4,
                            position_size=50000,
                        )
                        response = client.get("/positions")
                        assert response.status_code == 200
                        data = response.json()
                        assert isinstance(data, list)
                        assert all((isinstance(item, Position) for item in data))


@pytest.fixture
def sample_bot_data():
    return {
        "name": "LiquidatorBot1",
        "status": "active",
        "last_active": datetime(2022, 5, 15, 10, 0),
    }


def test_create_liquidator_bot(sample_bot_data):
    client = TestClient(app)
    response = client.post("/bots", json=sample_bot_data)
    assert response.status_code == 200

    def test_get_liquidator_bot():
        with mock.patch("main.app") as fastapi_app:
            fastapi_app.return_value = TestClient(app)
            bot_name = "LiquidatorBot1"
            response = client.get(f"/bots/{bot_name}")
            assert response.status_code == 200

            def test_update_liquidator_bot():
                with mock.patch("main.app") as fastapi_app:
                    fastapi_app.return_value = TestClient(app)
                    bot_name = "LiquidatorBot1"
                    response = client.put(f"/bots/{bot_name}")
                    assert response.status_code == 200


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

        def test_get_market_depth_endpoint(client):
            response = client.get("/market-depth")
            assert response.status_code == 200
            data = response.json()
            assert "ticker" in data
            assert "ask" in data
            assert "bid" in data


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

        def test_websocket_endpoint(client):
            response = client.get("/ws")
            assert response.status_code == 101
            assert response.headers["Connection"] == "Upgrade"

            def test_price_update_service():
                price_updates = []
                service = PriceUpdateService()
                with pytest.raises(AssertionError):
                    asyncio.get_event_loop().run_until_complete(
                        service.websocket_endpoint()
                    )

                    def mock_get_price_updates(
                        self, websocket: WebSocket, *args, **kwargs
                    ):
                        self.price_updates.append(
                            {"symbol": "BTC/USDT", "price": 40000}
                        )
                        return asyncio.Future()

                    service.get_price_updates = mock_get_price_updates
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(service.websocket_endpoint())
                    assert len(result) == 1
                    assert isinstance(result[0], dict)
                    assert result[0]["symbol"] == "BTC/USDT"
                    assert result[0]["price"] == 40000

                    def test_price_update_service_not_connected():
                        with pytest.raises(ConnectionError):
                            PriceUpdateService().websocket_endpoint()


@pytest.mark.use_testfiles
class TestUpdateUserPortfolio:

    @pytest.fixture
    def background_task(self):
        return BackgroundTasks()

    @pytest.mark.asyncio
    async def test_background_job_updates_user_portfolio_every_minute(
        self, background_task
    ):
        response = background_task.update_user_portfolio(1)
        assert response == {
            "message": "Background job to update user portfolio started."
        }
        time.sleep(60)
        portfolio_value = calculate_portfolio_value(1)
        assert portfolio_value == 1010


def test_generate_tax_report():
    with pytest.raises(HTTPException):
        if not TaxReportRequest.start_date or not TaxReportRequest.end_date:
            raise HTTPException(status_code=400, detail="Missing start or end dates.")
            request_data = TaxReportRequest(
                start_date="2023-01-01", end_date="2023-12-31"
            )
            tax_report = generate_tax_report(request_data)
            assert type(tax_report) == dict
            assert "result" in tax_report
            assert "report_data" in tax_report

            def test_create_tax_report():
                request_data = TaxReportRequest(
                    start_date="2023-01-01", end_date="2023-12-31"
                )
                with pytest.approx(
                    {
                        "result": "Tax Report Generated",
                        "report_data": [
                            {"date": "2023-01-01"},
                            {"date": "2023-02-01"},
                            ...,
                            {"date": "2023-12-31"},
                        ],
                    }
                ).item():
                    response = create_tax_report(request_data)
                    assert response.status_code == 200


@pytest.fixture
def risk_manager():
    return RiskManager()


def test_monitor_exposure(mocker):
    risk_manager = RiskManager()

    def mock_timestamp() -> datetime:
        return datetime.now()

    mocker.patch.object(risk_manager, "timestamp", new=mock_timestamp)
    exposure = Exposure(
        user_id=random.randint(1, 1000),
        limit=500.0,
        current_exposure=300.0,
        timestamp=datetime.now(),
    )
    with pytest.raises(HTTPException):
        risk_manager.monitor(exposure)

        def test_get_user_exposure():
            risk_manager = RiskManager()
            exposure = Exposure(
                user_id=1, limit=500.0, current_exposure=350.0, timestamp=datetime.now()
            )
            risk_manager.exposures.append(exposure)
            exposure_in_user_format = risk_manager.get_user_exposure(1)
            assert exposure_in_user_format == exposure

            def test_monitor_and_get_user_exposure():
                risk_manager = RiskManager()
                exposure = Exposure(
                    user_id=1,
                    limit=500.0,
                    current_exposure=350.0,
                    timestamp=datetime.now(),
                )
                with pytest.raises(HTTPException):
                    risk_manager.monitor(exposure)
                    with pytest.raises(HTTPException):
                        risk_manager.get_user_exposure(1)

                        @pytest.mark.parametrize(
                            "exposure, expected",
                            [
                                (
                                    Exposure(
                                        user_id=2,
                                        limit=1000.0,
                                        current_exposure=500.0,
                                        timestamp=datetime.now(),
                                    ),
                                    400,
                                ),
                                (
                                    Exposure(
                                        user_id=2,
                                        limit=1000.0,
                                        current_exposure=999.0,
                                        timestamp=datetime.now(),
                                    ),
                                    200,
                                ),
                            ],
                        )
                        def test_risk_manager_exceptions(risk_manager, mocker):
                            risk_manager.exposures = []
                            mock_timestamp = mocker.patch.object(
                                risk_manager, "timestamp", new=mock_timestamp
                            )
                            for exposure, expected_status in exposure_cases:
                                with pytest.raises(HTTPException) as e_info:
                                    risk_manager.monitor(exposure)
                                    assert e_info.value.status_code == expected_status

                                    @pytest.mark.parametrize(
                                        "exposure, user_exposure",
                                        [
                                            (
                                                Exposure(
                                                    user_id=2,
                                                    limit=1000.0,
                                                    current_exposure=500.0,
                                                    timestamp=datetime.now(),
                                                ),
                                                1,
                                            )
                                        ],
                                    )
                                    def test_risk_manager_get_user_exposure(
                                        risk_manager, mocker
                                    ):
                                        risk_manager.exposures = []
                                        mock_timestamp = mocker.patch.object(
                                            risk_manager,
                                            "timestamp",
                                            new=mock_timestamp,
                                        )
                                        exposure = Exposure(
                                            user_id=2,
                                            limit=1000.0,
                                            current_exposure=500.0,
                                            timestamp=datetime.now(),
                                        )
                                        with pytest.raises(HTTPException):
                                            risk_manager.monitor(exposure)
                                            with pytest.raises(HTTPException) as e_info:
                                                exposure_in_user_format = (
                                                    risk_manager.get_user_exposure(2)
                                                )
                                                assert e_info.value.status_code == 200
                                                assert (
                                                    exposure_in_user_format.user_id == 2
                                                )
                                                assert (
                                                    exposure_in_user_format.current_exposure
                                                    == 500.0
                                                )


def test_request_signatures():
    client = TestClient(app)
    signature_request_data = {
        "address": "test_address",
        "required_signatures": 2,
        "signatures": [],
    }
    response = client.post("/request_signatures", json=signature_request_data)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data and "address" in data

    def test_approve_signatures():
        client = TestClient(app)
        signature_request_data = {
            "address": "test_address",
            "required_signatures": 2,
            "signatures": ["sig1", "sig2"],
        }
        response = client.post("/request_signatures", json=signature_request_data)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data and "address" in data

        def test_approve_signatures_with_insufficient_signatures():
            client = TestClient(app)
            with pytest.raises(HTTPException):
                signature_request_data = {
                    "address": "test_address",
                    "required_signatures": 2,
                    "signatures": ["sig1"],
                }
                response = client.post(
                    "/request_signatures", json=signature_request_data
                )
                assert response.status_code == 400
                data = response.json()
                assert (
                    "detail" in data
                    and data["detail"] == "Not enough signatures received."
                )


@pytest.mark.parametrize(
    "account_data, expected",
    [
        (
            {
                "account_id": "acc123",
                "source_account_margin": 1000.0,
                "target_account_margin": 500.0,
            },
            "Cross-margin position transfer between accounts was successful.",
        ),
        (
            {
                "account_id": "invalid_acc",
                "source_account_margin": -50.0,
                "target_account_margin": 200.0,
            },
            "Invalid margin values",
        ),
    ],
)
def test_transfer_cross_margin_positions(account_data, expected):
    with pytest.raises(HTTPException) as e:
        response = app.test_client().post(
            "/transfer_cross_margin_positions", json=account_data
        )
        assert e.type == HTTPException
        assert response.status_code == 200
        assert response.json() == {"message": expected}


@pytest.fixture()
def client():
    with TestClient(app) as tc:
        yield tc

        def test_create_amm_pair(client):
            amm_pair = AMMPair(token0="BTC", token1="USDT")
            response = client.post("/amm_pairs", json=amm_pair.dict())
            assert response.status_code == 200
            assert "id" in response.json()

            def test_create_invalid_amm_pair(client):
                with pytest.raises(HTTPException):
                    amm_pair_data = AMMPair(token0="invalid_token", token1="USDT")
                    client.post("/amm_pairs", json=amm_pair_data.dict())


def test_create_otc_quote_request():
    client = TestClient(app)
    req_data = {
        "request_id": "1234",
        "trader_name": "Trader One",
        "instrument_type": "STK",
        "amount": 100.0,
        "settlement_date": datetime.now().date(),
    }
    response = client.post("/otc_quote_request", json=req_data)
    assert response.status_code == 200
    assert "desk_id" in response.json()


def test_encrypt_decrypt():
    private_key = b"my_private_key"
    bridge = LiquidityBridge("public_key", private_key)
    encrypted_data = bridge.encrypt(b"data_to_be_encrypted")
    decrypted_data = bridge.decrypt(base64.b64encode(encrypted_data).decode())
    assert len(decrypted_data) == len(b"data_to_be_encrypted")
    assert decrypted_data == b"data_to_be_encrypted"

    def test_add_bridge():
        manager = LiquidityBridgeManager()
        manager.add_bridge(public_key="public1", private_key="private1")
        manager.add_bridge(public_key="public2", private_key="private2")
        bridge1 = LiquidityBridge("public1", "private1")
        bridge2 = LiquidityBridge("public2", "private2")
        assert manager.get_bridge_by_public_key("public1") == bridge1
        assert manager.get_bridge_by_public_key("public2") == bridge2

        def test_get_bridge_by_public_key():
            manager = LiquidityBridgeManager()
            manager.add_bridge(public_key="public1", private_key="private1")
            with pytest.raises(HTTPException) as exc:
                manager.get_bridge_by_public_key("invalid_public_key")
                assert str(exc.value) == "Bridge not found"


def test_endpoint_not_found():
    client = TestClient(app)
    response = client.get("/notfoundendpoint")
    assert response.status_code == 404
    assert "Unsupported cross-chain bridge." in str(response.content)

    def test_bridge_tokens_success():
        client = TestClient(app)
        request_data = BridgingRequest(
            id=str(uuid.uuid4()),
            from_chain="Ethereum",
            to_chain="Solana",
            amount=10,
            destination_address="testaddress",
        )
        response = client.post("/bridging", json=request_data.dict())
        assert response.status_code == 200
        assert "success" in str(response.content)
        assert "Bridging operation for" in str(response.content)

        def test_bridge_tokens_unsupported_chain():
            client = TestClient(app)
            with pytest.raises(HTTPException):
                request_data = BridgingRequest(
                    id=str(uuid.uuid4()),
                    from_chain="UnsupportedChain",
                    to_chain="Solana",
                    amount=10,
                    destination_address="testaddress",
                )
                response = client.post("/bridging", json=request_data.dict())
                assert response.status_code == 404
                assert "Unsupported cross-chain bridge." in str(response.content)


@pytest.fixture
def client():
    with TestClient(app) as ac:
        yield ac

        def test_bulk_order_import(client):
            response = client.post(
                "/bulk_order_import/", files={"file": open("test_orders.csv", "rb")}
            )
            assert response.status_code == 200
            assert (
                response.json()["message"] == "CSV file has been successfully imported."
            )

            def test_bulk_order_import_failed_client():
                with pytest.raises(AssertionError):
                    client = TestClient(app)
                    response = client.post(
                        "/bulk_order_import/",
                        files={"file": open("non_existent_orders.csv", "rb")},
                    )
                    assert response.status_code == 200
                    assert (
                        response.json()["message"]
                        == "CSV file has been successfully imported."
                    )


def test_create_netting_group():
    new_netting_group = NettingGroup(id=1, name="Alpha", member_accounts=["1234"])
    response = create_netting_group(new_netting_group)
    assert response == {"message": "Netting group created successfully"}

    def test_get_netting_group():
        alpha_netting_group = NettingGroup(id=1, name="Alpha", member_accounts=["1234"])
        netting_groups.append(alpha_netting_group)
        response = get_netting_group(1)
        assert response == alpha_netting_group

        def test_netting_group_not_found():
            with pytest.raises(HTTPException):
                get_netting_group(999)


@pytest.fixture
def sample_bridge():
    return Bridge(src_mac="00:11:22:33:44", dst_bridge_id="12345678")


def test_create_bridges(sample_bridge):
    assert sample_bridge.src_mac == "00:11:22:33:44"
    assert sample_bridge.dst_bridge_id is None
    assert sample_bridge.status == "inactive"
    response = Bridge(src_mac="00:11:22:33:45", dst_bridge_id="12345679").app
    assert response.src_mac == "00:11:22:33:45"
    assert response.dst_bridge_id == "12345679"
    assert response.status == "inactive"

    def test_get_bridge(sample_bridge):
        with pytest.raises(HTTPException) as ex:
            Bridge().get_bridge(id="12345678")
            assert str(ex.value).startswith("Bridge ID not found.")

            def test_post_bridges_with_existing_id():
                with pytest.raises(HTTPException) as ex:
                    sample_bridge = Bridge(
                        src_mac="00:11:22:33:45", dst_bridge_id="12345679"
                    )
                    response = sample_bridge.app.post("/bridge", data=sample_bridge)
                    assert str(ex.value).startswith("Bridge ID already exists.")

                    def test_post_bridges_success():
                        with pytest.raises(HTTPException) as ex:
                            sample_bridge = Bridge(
                                src_mac="00:11:22:33:46", dst_bridge_id="12345678"
                            )
                            response = sample_bridge.app.post(
                                "/bridge", data=sample_bridge
                            )
                            assert str(ex.value).startswith("New bridge created")


@pytest.mark.parametrize(
    "token0, token1, expected",
    [
        (
            {"id": 1},
            {"id": 2},
            {
                "token0": {
                    "id": 1,
                    "amount_shares": 500,
                    "liquidity_pool_address": "0x...",
                },
                "token1": {
                    "id": 2,
                    "amount_shares": 200,
                    "liquidity_pool_address": "0x...",
                },
            },
        ),
        (
            {"id": 3},
            {"id": 4},
            {
                "token0": {
                    "id": 3,
                    "amount_shares": 300,
                    "liquidity_pool_address": "0x...",
                },
                "token1": {
                    "id": 4,
                    "amount_shares": 100,
                    "liquidity_pool_address": "0x...",
                },
            },
        ),
    ],
)
def test_get_concentrated_liquidity(token0, token1, expected):
    client = TestClient(app)
    response = client.get("/amm/concentrated-liquidity")
    assert response.status_code == 200
    data = json.loads(response.text)
    assert data["token0"]["id"] == token0["id"]
    assert data["token0"]["amount_shares"] == token0["amount_shares"]
    assert data["token1"]["id"] == token1["id"]
    assert data["token1"]["amount_shares"] == token1["amount_shares"]


def test_verify_state():
    state = CrossChainState(
        chain_id="test_chain",
        sender="0x1234567890123456789012345678901234",
        recipient="0x0987654321098765432109876543210987",
        amount=100,
        timestamp=datetime.utcnow(),
    )
    assert CrossChainState.verify_state(state) == True

    def test_verify_and_process_state():
        state = CrossChainState(
            chain_id="test_chain",
            sender="0x1234567890123456789012345678901234",
            recipient="0x0987654321098765432109876543210987",
            amount=100,
            timestamp=datetime.utcnow(),
        )
        assert CrossChainState.verify_state(state) == True
        with pytest.raises(HTTPException):
            with pytest.delegates() as delegates:
                CrossChainState.verify_and_process_state(state)
                raise Exception("Test exception")


def test_validate_compliance_attestation():
    attestation = ComplianceAttestation(
        attester_public_key="attester_public_key",
        timestamp=datetime.now(),
        attestation_data="valid_data",
    )
    assert validate_compliance_attestation(attestation) == True

    def test_attest_compliance_attestation_valid():
        attestation = ComplianceAttestation(
            attester_public_key="attester_public_key",
            timestamp=datetime.now(),
            attestation_data="valid_data",
        )
        with pytest.raises(HTTPException):
            assert attest_compliance_attestation(attestation) == True

            def test_attest_compliance_attestation_invalid():
                attestation = ComplianceAttestation(
                    attester_public_key="attester_public_key",
                    timestamp=datetime.now(),
                    attestation_data="invalid_data",
                )
                with pytest.raises(HTTPException):
                    assert attest_compliance_attestation(attestation) == False


def test_calculate_rebalance_ratio():
    assert calculate_rebalance_ratio(10, 20, 30) == 0.3333333333333
    assert calculate_rebalance_ratio(100, 500, 600) == 1 / 6

    def test_startup_function():
        with pytest.raises(asyncio.CancelledError):
            startup()

            def test_liquidity_pool_rebalance_endpoint(client: TestClient):
                response = client.get("/rebalance")
                assert response.status_code == 200


app = FastAPI()


def test_position_netting():
    position_netting = CrossMarginPositionNetting()
    positions_dict = {
        "member1": {"net_position": 1000},
        "member2": {"net_position": -500},
    }
    with pytest.raises(ValueError):
        position_netting.create_netting_entry(
            margin_member="new_member", net_position=200
        )
        position_netting.positions["member1"]["net_position"] = 500
        with pytest.raises(ValueError):
            position_netting.update_netting_entry(
                margin_member="member1", new_net_position=-300
            )
            positions_dict = position_netting.positions_dict
            assert len(positions_dict) == 2
            client = TestClient(app)
            response = client.get("/positions")
            assert response.status_code == 200
            assert response.json() == positions_dict

            def test_endpoint():
                client = TestClient(app)
                response = client.get("/endpoint")
                assert response.status_code == 200


@pytest.main
def test_get_positions():
    client = TestClient(app)
    response = client.get("/positions")
    assert response.status_code == 200
    assert isinstance(response.json(), DebtPositions)

    def test_create_position():
        client = TestClient(app)
        new_debt_position = DebtPosition(
            id=str(uuid.uuid4()),
            creditor_id="C1",
            debtor_id="D1",
            amount=1000.0,
            interest_rate=5.0,
            maturity_date=datetime(2023, 12, 10),
        )
        response = client.post("/positions", json=new_debt_position.dict())
        assert response.status_code == 200
        assert "id" in response.json()

        def test_update_position():
            client = TestClient(app)
            updated_debt_position = DebtPosition(
                id=str(uuid.uuid4()),
                creditor_id="C1",
                debtor_id="D1",
                amount=1000.0,
                interest_rate=5.0,
                maturity_date=datetime(2023, 12, 10),
            )
            response = client.put(
                "/positions/{position_id}".format(position_id=str(uuid.uuid4())),
                json=updated_debt_position.dict(),
            )
            assert response.status_code == 200
            assert "id" in response.json()

            def test_update_position_not_found():
                client = TestClient(app)
                with pytest.raises(HTTPException):
                    response = client.put(
                        "/positions/{position_id}".format(position_id="nonexistent"),
                        json={"creditor_id": "C1", "debtor_id": "D1"},
                    )
                    assert response.status_code == 404
                    assert "Position not found." in str(response.content)


def test_create_position():
    client = TestClient(app)
    position_data = CollateralPosition(
        position_id=str(uuid.uuid4()),
        asset="Asset",
        protocol="Protocol",
        amount=100.0,
        timestamp=datetime.now(),
    )
    response = client.post("/positions", json=position_data.dict())
    assert response.status_code == 200
    assert "result" in response.json()
    assert "position" in response.json()

    def test_update_position():
        client = TestClient(app)
        position_data = CollateralPosition(
            position_id=str(uuid.uuid4()),
            asset="Asset",
            protocol="Protocol",
            amount=100.0,
            timestamp=datetime.now(),
        )
        response = client.post("/positions", json=position_data.dict())
        assert response.status_code == 200
        new_position_data = CollateralPosition(
            position_id=str(uuid.uuid4()),
            asset="Asset",
            protocol="Protocol",
            amount=200.0,
        )
        response = client.put(
            f"/positions/{position_data.position_id}", json=new_position_data.dict()
        )
        assert response.status_code == 200
        assert "result" in response.json()
        assert "updated_position" in response.json()

        def test_get_position_not_found():
            client = TestClient(app)
            response = client.get("/positions/nonexistingpositionid")
            assert response.status_code == 404
            assert "detail" in response.json()
            assert "Collateral Position not found" == response.json()["detail"]


@pytest.fixture
def options_hedge_endpoint():
    with mock.patch("requests.get") as mock_get:
        yield mock_get

        def test_options_hedge_success(options_hedge_endpoint):
            mock_get.return_value.json.return_value = {"delta": 0.1, "price": 50}
            symbol = "AAPL"
            strike_price = 100
            expiration_date = datetime(2023, 12, 15)
            quantity = 10
            result = options_hedge_endpoint("AAPL", 100, "2023-12-15", 10)
            assert result == {
                "delta": 0.1,
                "price": 50,
                "stock_symbol": "AAPL",
                "strike_price": 100,
                "expiration_date": datetime(2023, 12, 15),
                "quantity": 10,
            }

            def test_options_hedge_failure():
                symbol = "INVALID_SYMBOL"
                strike_price = 100
                expiration_date = datetime(2024, 1, 5)
                quantity = 10
                with pytest.raises(HTTPException) as exc:
                    options_hedge_endpoint("INVALID_SYMBOL", 100, "2024-01-05", 10)
                    assert str(exc.value) == "Server error while fetching option data."

                    def test_options_hedge_endpoint():
                        app = FastAPI()

                        @app.post("/options_hedge")
                        async def options_hedge(
                            symbol: str,
                            strike_price: float,
                            expiration_date: datetime,
                            quantity: int,
                        ):
                            if symbol not in ["AAPL", "GOOGL"]:
                                raise HTTPException(
                                    status_code=400, detail="Invalid stock symbol."
                                )
                                return {
                                    "stock_symbol": symbol,
                                    "strike_price": strike_price,
                                    "expiration_date": expiration_date,
                                    "quantity": quantity,
                                }
                            client = TestClient(app)
                            test_options_hedge_success(client)
                            test_options_hedge_failure(client)


@pytest.fixture
def new_rotation():
    return CustodyRotation()


def test_create_rotation_event():
    event_id = "rotation1"
    start_date = datetime(2023, 5, 15)
    end_date = datetime(2023, 6, 15)
    response = app.test_client().post(
        "/rotation",
        json={
            "custodian": "ABC Corp",
            "event_id": event_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "rotation_id" in data

    def test_last_rotation_event():
        rotation_events = [
            {
                "event_id": "rotation1",
                "start_date": "2023-05-15",
                "end_date": "2023-06-15",
                "custodian": "ABC Corp",
            },
            {
                "event_id": "rotation2",
                "start_date": "2023-07-10",
                "end_date": "2023-08-10",
                "custodian": "DEF Ltd",
            },
        ]
        rotation = CustodyRotation()
        rotation.rotation_events = rotation_events
        assert rotation.last_rotation_event() == {
            "event_id": "rotation2",
            "start_date": "2023-07-10",
            "end_date": "2023-08-10",
            "custodian": "DEF Ltd",
        }

        def test_add_rotation_event():
            new_rotation = CustodyRotation()
            event_id = "rotation1"
            start_date = datetime(2023, 5, 15)
            end_date = datetime(2023, 6, 15)
            assert new_rotation.last_rotation_event() is None
            new_rotation.add_rotation_event(event_id, start_date, end_date, "ABC Corp")
            assert new_rotation.last_rotation_event().get("event_id") == "rotation1"
            assert (
                newRotation.get_rotation_events()[-1].get("start_date")
                == start_date.isoformat()
            )


@pytest.fixture
def client():
    with TestClient(app) as FastAPIClient:
        yield FastAPIClient

        def test_cancel_order(client):
            response = client.get("/cancel/1")
            assert response.status_code == 200
            assert "message" in response.json()
            assert "order_id" in response.json()
            response = client.get("/cancel/9999")
            assert response.status_code == 404
            assert "Order not found" in response.text


@pytest.fixture
def client():
    yield TestClient(app)

    def test_migrate(client):
        response = client.get("/migrate")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


def test_audit_log_endpoint():
    client = TestClient(app)
    response = client.get("/audit-log")
    assert response.status_code == 200
    assert len(response.json()["audits"]) >= 5
    for audit in response.json()["audits"]:
        assert isinstance(audit["timestamp"], datetime.datetime)
        assert isinstance(audit["event_type"], str)
        assert isinstance(audit["user_id"], int)
        assert isinstance(audit["description"], str)


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        @pytest.mark.asyncio
        async def test_stop_loss_order(client):
            data = StopLossOrder(
                symbol="AAPL",
                quantity=100,
                price=150,
                stop_loss_price=None,
                trailing_stop_distance=20.0,
            )
            response = await client.post("/stop-loss-order", json=data)
            assert response.status_code == 200

            def test_trailing_stop(client):
                data = StopLossOrder(
                    symbol="AAPL",
                    quantity=100,
                    price=150,
                    stop_loss_price=160,
                    trailing_stop_distance=10.0,
                )
                response = client.get(
                    "/stop-loss-order-trailing",
                    params={"order_id": "test_order"},
                    json=data,
                )
                result = response.json()
                assert "trailing_stop_price" in result


@pytest.fixture()
def client():
    with TestClient(app) as tc:
        yield tc

        def test_set_trading_parameters(client):
            response = client.post(
                "/parameters", json={"risk_per_trade": 0.7, "trade_frequency": 15}
            )
            assert response.status_code == 200
            data = response.json()
            assert "status" in data and data["status"] == "trading_parameters_updated"

            def test_get_trading_parameters(client):
                response = client.get("/parameters")
                assert response.status_code == 200
                data = response.json()
                assert "risk_per_trade" in data and "trade_frequency" in data


@pytest.fixture
def test_csv():
    data = b"header1,header2,order_id\nvalue1,value2,123\nvalue3,value4,456"
    return IO[data]


def test_bulk_order_import(test_csv):
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file_path = os.path.join(tmpdirname, "bulk_order_import.csv")
        with open(temp_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["header1", "header2", "order_id"])
            writer.writerow(["value1", "value2", "123"])
            writer.writerow(["value3", "value4", "456"])
            response = bulk_order_import(
                files=[File(io=temp_file_path, filename="bulk_order_import.csv")]
            )
            assert response.status_code == 200
            assert os.path.exists(
                os.path.join(tmpdirname, "bulk_order_data", "bulk_order_import.csv")
            )


importpytest


def test_generate_compliance_report():
    report_data = generate_compliance_report()
    assert isinstance(report_data, dict)
    assert "report_due_date" in report_data

    @pytest.mark.asyncio
    async def test_compliance_report_endpoint():
        client = TestClient(app)
        response = await client.get("/compliance-report")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)

        @pytest.mark.asyncio
        async def test_schedule_compliance_report_endpoint():
            client = TestClient(app)
            report_data = {"report_due_date": "2023-01-20T12:00:00Z"}
            response = await client.post(
                "/schedule-compliance-report", json=report_data
            )
            assert response.status_code == 200
            assert isinstance(response.json(), dict)

            @pytest.mark.asyncio
            async def test_endpoint():
                client = TestClient(app)
                with pytest.raises(asyncio.TimeoutError):
                    response = await client.get("/endpoint")
                    assert (
                        False
                    ), f"Expected asyncio timeout error but got status code {response.status_code}"


@pytest.fixture()
def client():
    with TestClient(app) as tc:
        yield tc

        def test_create_identity(client):
            identity_data = {"user_id": "test_user1", "public_key": "random_public_key"}
            response = client.post("/identity", json=identity_data)
            assert response.status_code == 200
            assert response.json() == {
                "user_id": "test_user1",
                "public_key": "random_public_key",
            }

            def test_get_identity(client):
                identity_data = {
                    "user_id": "test_user2",
                    "public_key": "random_public_key",
                }
                response = client.get("/identity/test_user2")
                assert response.status_code == 200
                assert response.json() == {
                    "user_id": "test_user2",
                    "public_key": "random_public_key",
                }

                def test_update_identity(client):
                    user_id = "test_user3"
                    updated_identity_data = {
                        "user_id": user_id,
                        "public_key": "new_random_public_key",
                    }
                    response = client.put(
                        "/identity/{user_id}".format(user_id=user_id),
                        json=updated_identity_data,
                    )
                    assert response.status_code == 200
                    assert response.json() == {
                        "user_id": user_id,
                        "public_key": "new_random_public_key",
                    }

                    def test_delete_identity(client):
                        user_id = "test_user4"
                        response = client.delete(
                            "/identity/{user_id}".format(user_id=user_id)
                        )
                        assert response.status_code == 200
                        assert response.json() == {
                            "detail": f"User ID {user_id} has been successfully deleted."
                        }


def test_list_collaterals():
    client = TestClient(app)
    response = client.get("/collaterals")
    assert response.status_code == 200
    data = response.json()
    for collateral in data:
        assert isinstance(collateral, dict)
        assert "chain" in collateral
        assert "token_address" in collateral
        assert "token_symbol" in collateral
        assert "amount" in collateral

        def test_deposit_collateral():
            client = TestClient(app)
            response = client.post(
                "/deposit-collateral",
                json={
                    "chain": "Binance Smart Chain",
                    "token_symbol": "DAI",
                    "amount": "5000000",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
            assert "chain" in data
            assert "token_address" in data
            assert "token_symbol" in data
            assert "amount" in data

            def test_request_collateral():
                client = TestClient(app)
                response = client.post(
                    "/request-collateral",
                    json={
                        "chain": "Ethereum",
                        "token_symbol": "DAI",
                        "amount": "5000000",
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, dict)
                assert "chain" in data
                assert "token_address" in data
                assert "token_symbol" in data
                assert "amount" in data


def test_get_all_pools():
    client = TestClient(app)
    response = client.get("/liquidity-pools")
    assert response.status_code == 200
    data = response.json()
    for item in data:
        assert isinstance(item, LiquidityPool)

        def test_validate_pool():
            pool1 = LiquidityPool(
                chain="ETH", token1="DAI", token2="USDC", liquidity=100
            )
            pool2 = LiquidityPool(chain="BTC", token1="BCH", token2="LTC", liquidity=50)
            system = LiquidityAggregationSystem()
            assert system.validate_pool(pool1)
            assert system.validate_pool(pool2) is False

            def test_add_pool():
                pool = LiquidityPool(
                    chain="ETH", token1="DAI", token2="USDC", liquidity=100
                )
                system = LiquidityAggregationSystem()
                assert system.add_pool(pool)


def test_generate_stress_test_data():
    data = generate_stress_test_data()
    assert isinstance(data, list)
    assert len(data) > 0

    def test_smart_contract_stress_test_endpoint():
        client = TestClient(app)
        response = client.get("/smart-contract-stress-test")
        assert response.request.method == "GET"
        assert response.status_code == 200
        content = response.json()
        data = content["data"]
        assert isinstance(data, list)
        assert len(data) > 0


@pytestmark
class TestMigration:

    def setup_method(self):
        self.migration_request = MigrationRequest(
            source_pool_id=str(uuid.uuid4()), target_pool_id="target_pool_id"
        )

        def test_migration_source_target_different(self):
            with pytest.raises(HTTPException) as e:
                router.post("/migration", request=self.migration_request)
                assert str(e.value) == "Source and target pool IDs must be different."

                def test_migration_no_error(self):
                    response = router.post("/migration", request=self.migration_request)
                    assert response.status_code == 200


def create_app():
    return app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as C:
        yield C

        def test_generate_claim(client):
            response = client.get(
                "/insurance-claim/generate", params={"policy_number": 123}
            )
            assert response.status_code == 200
            data = response.json()
            expected_data = {
                "claim_id": str(uuid.uuid4()),
                "policy_number": 123,
                "claim_amount": 0.0,
                "claim_status": "pending",
                "user_id": None,
                "date_claimed": datetime.now(),
            }
            assert data == expected_data

            def test_process_claim(client):
                response = client.get(
                    "/insurance-claim/process",
                    params={"claim_id": "f2e4c1b-5d3a-4f8b-a9ab-fdfc0b1d9c6"},
                )
                data = response.json()
                assert response.status_code == 200
                expected_data = {
                    "claim_id": "f2e4c1b-5d3a-4f8b-a9ab-fdfc0b1d9c6",
                    "policy_number": None,
                    "claim_amount": 500.0,
                    "claim_status": "approved",
                    "user_id": None,
                    "date_claimed": datetime(2022, 12, 10, 15, 30),
                }
                assert data == expected_data

                def test_invalid_claim(client):
                    response = client.get(
                        "/insurance-claim/process",
                        params={"claim_id": "f2e4c1b-5d3a-4f8b-a9ab-fdfc0b1d9c6"},
                    )
                    data = response.json()
                    assert response.status_code == 400
                    expected_data = {"detail": "Invalid Claim Status", "code": 400}
                    assert data == expected_data

                    def test_claim_status(client):
                        claims = [
                            InsuranceClaim.generate_claim(123),
                            InsuranceClaim.generate_claim(456),
                        ]
                        response = client.post(
                            "/insurance-claim", json=claims[0]._asdict()
                        )
                        data = response.json()
                        assert response.status_code == 200
                        expected_data = {
                            "claim_id": str(uuid.uuid4()),
                            "policy_number": claims[0].policy_number,
                            "claim_amount": claims[0].claim_amount,
                            "claim_status": claims[0].claim_status,
                            "user_id": None,
                            "date_claimed": datetime.now(),
                        }
                        assert data == expected_data
                        response = client.post(
                            "/insurance-claim", json=claims[1]._asdict()
                        )
                        data = response.json()
                        assert response.status_code == 200
                        expected_data = {
                            "claim_id": str(uuid.uuid4()),
                            "policy_number": claims[1].policy_number,
                            "claim_amount": claims[1].claim_amount,
                            "claim_status": claims[1].claim_status,
                            "user_id": None,
                            "date_claimed": datetime.now(),
                        }
                        assert data == expected_data


@pytest.fixture
def client():
    yield TestClient(app)

    def test_get_risk_factors(client):
        response = client.get("/risk_factors")
        assert response.status_code == 200
        result = response.json()
        risk_factors = result["riskFactors"]
        assert len(risk_factors) > 0


@pytest.fixture()
def client():
    with TestClient(app) as TC:
        yield TC

        def test_withdraw(client):
            request_data = WithdrawalRequest(
                destination_address="123 Main St", amount=500.0
            )
            response = client.post("/withdraw", json=request_data.dict())
            assert response.status_code == 200
            result = response.json()
            assert "message" in result
            assert "amount" in result
            assert "balance" in result

            def test_withdraw_insufficient_balance(client):
                request_data = WithdrawalRequest(
                    destination_address="123 Main St", amount=1500.0
                )
                response = client.post("/withdraw", json=request_data.dict())
                assert response.status_code == 400
                result = response.json()
                assert "detail" in result


@pytest.fixture()
def client():
    return TestClient(app)


def test_add_audit_log(client):
    audit_log_data = {
        "timestamp": str(time.time()),
        "user_id": 123,
        "action": "completed",
        "data": {"result": "success"},
    }
    response = client.post("/audit-log", json=audit_log_data)
    assert response.status_code == 200
    assert AuditLog.AUDIT_LOGS[0].data["result"] == "success"


def test_enter_leave_maintenance_mode():
    client = TestClient(app)
    response = client.get("/maintenance")
    assert response.status_code == 200
    content = response.json()
    assert content["mode"] is True
    assert asyncio.Lock().acquired() is True

    async def _leave_mode():
        return await MaintenanceMode().leave_mode()

    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(_leave_mode())
    assert response.status_code == 200
    content = response.json()
    assert "mode" not in content or content["mode"] is False


def test_calculate_transaction_costs():
    trcs = TransactionCosts(volume=1000, spread_percentage=2, base_fee=0.01)
    expected_output = {"total_cost": 11.1, "base_fee": 0.01, "spread_cost": 11.1 - 0.01}
    result = calculate_transaction_costs(trcs)
    assert result == expected_output

    def test_get_transaction_costs():
        trcs = TransactionCosts(volume=1000, spread_percentage=2, base_fee=0.01)
        client = TestClient(app)
        response = client.post("/transaction-costs", json=trcs.json())
        result = response.json()
        assert response.status_code == 200
        assert "total_cost" in result
        assert "base_fee" in result
        assert "spread_cost" in result


def test_invalid_request_id():
    with pytest.raises(HTTPException):
        otc_router.quote_request("invalid_req")

        def test_quote_request():
            req = QuoteRequest(request_id="req1", trader_name="TraderA")
            assert req.request_id == "req1"
            assert req.trader_name == "TraderA"

            def test_settlement():
                settlement = QuoteRequest(request_id="req2", trader_name="TraderB")
                with pytest.raises(HTTPException):
                    otc_router.settle(settlement)


def test_endpoint():
    client = TestClient(app)
    response = client.get("/endpoint")
    assert response.status_code == 200

    def test_wash_trading():
        client = TestClient(app)
        user_id_1 = 1
        user_id_2 = 2
        timestamp = datetime.datetime(2023, 5, 10, 10)
        amount_1 = 100.0
        amount_2 = 50.0
        response = client.post(
            "/trade",
            json={"timestamp": timestamp, "user_id": user_id_1, "amount": amount_1},
        )
        assert response.status_code == 200
        trade_1 = response.json()
        response = client.post(
            "/trade",
            json={"timestamp": timestamp, "user_id": user_id_2, "amount": amount_2},
        )
        assert response.status_code == 200
        trade_2 = response.json()
        trades = client.get(f"/user_trades/{user_id_1}")
        assert trades.status_code == 200
        user_trades = trades.json()
        assert len(user_trades) > 0
        for trade in user_trades:
            assert trade["amount"] != amount_2


@pytest.mark.asyncio
async def test_is_delisted_returns_false():
    trading_pair = TradingPair("BTC-USD", "BTC/USD")
    assert trading_pair.is_delisted() is False

    @pytest.mark.asyncio
    async def test_delist_trading_pairs_not_all_delisted():
        trading_pairs = [
            TradingPair("BTC-USD", "BTC/USD"),
            TradingPair("ETH-BTC", "ETH/BTC"),
        ]

        async def mock_system():
            return AsyncMock()

        system_mock = mock.System()
        trading_pair1 = MagicMock()
        trading_pair2 = MagicMock()
        trading_pairs[0] = trading_pair1
        trading_pairs[1] = trading_pair2
        with pytest.raises(AsyncIOError):
            await delist_trading_pairs(system_mock)


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_generate_signal(client):
            response = client.post("/generate-signal")
            assert response.status_code == 200
            data = response.json()
            assert (
                "status" in data and data["status"] == "signals generated successfully"
            )


def test_vesting_schedule_creation():
    start_date = date(2023, 1, 1)
    total_tokens = 1000000
    schedule = VestingSchedule(start_date, total_tokens)
    assert schedule.start_date == start_date
    assert schedule.total_tokens == total_tokens
    assert schedule.vested_tokens == 0

    def test_vesting_schedule_vesting():
        start_date = date(2023, 1, 1)
        total_tokens = 1000000
        vesting_amount = 50000
        schedule = VestingSchedule(start_date, total_tokens)
        with pytest.raises(HTTPException) as exc:
            schedule.vest(vesting_amount)
            assert str(exc.value) == "Insufficient tokens to vest."

            def test_vesting_schedule_vesting_valid():
                start_date = date(2023, 1, 1)
                total_tokens = 1000000
                vesting_amount = 50000
                schedule = VestingSchedule(start_date, total_tokens)
                with patch("main.date") as mock_date:
                    mock_date.today.return_value = start_date - timedelta(days=10)
                    response = schedule.vest(vesting_amount)
                    assert response.status_code == 200
                    assert response.json() == {
                        "message": f"Vested {vesting_amount} tokens."
                    }


@pytest.fixture()
def did_manager():
    return DIDManager()


def test_create_did(did_manager):
    new_did = uuid.uuid4().hex
    expected_did_doc = DIDDocument(
        did=new_did,
        verificationMethod="http://example.com/verification",
        authenticationMethod="http://example.com/authentication",
    )
    assert did_manager.dids == {}
    response = did_manager.create_did(expected_did_doc)
    assert response == new_did
    assert did_manager.dids != {}

    def test_get_did(did_manager):
        new_did = uuid.uuid4().hex
        expected_did_doc = DIDDocument(
            did=new_did,
            verificationMethod="http://example.com/verification",
            authenticationMethod="http://example.com/authentication",
        )
        response = did_manager.create_did(expected_did_doc)
        assert response == new_did
        response = did_manager.get_did(new_did)
        expected_response = DIDDocument(
            did=new_did,
            verificationMethod="http://example.com/verification",
            authenticationMethod="http://example.com/authentication",
        )
        assert response == expected_response


@pytest.fixture
def identity_app():
    app = DecentralizedIdentityRecoveryApp()
    yield app
    app.client.close()

    def test_get_identity(identity_app):
        identity_id = str(uuid.uuid4())
        identity = Identity(id=identity_id)
        identity_app.identity_store[identity_id] = identity
        response = identity_app.get_identity(identity_id)
        assert response == identity

        def test_post_recovery_question_answer(identity_app):
            identity_id = str(uuid.uuid4())
            email = "testuser@example.com"
            recovery_question = "What is your favorite color?"
            recovery_answer = "Blue"
            identity = Identity(id=identity_id, email=email)
            identity_app.identity_store[identity.id] = identity
            with pytest.raises(HTTPException):
                identity_app.post_recovery_question_answer(
                    identity_id, recovery_question, recovery_answer
                )
                recovery_identity = identity_app.post_recovery_question_answer(
                    identity_id, recovery_question, recovery_answer
                )
                assert (
                    recovery_identity.recovery_question
                    == "What is your favorite color?"
                )
                assert recovery_identity.recovery_answer == "Blue"
                assert recovery_identity.id != identity_id


def test_provide_liquidity():
    client = TestClient(app)
    response = client.post("/provide-liquidity", json={"amount": 1000.0})
    assert response.status_code == 200
    assert "timestamp" in response.json()
    assert "amount_provided" in response.json()
    response = client.post("/provide-liquidity", json={"amount": -500.0})
    assert response.status_code == 400
    assert "Amount must be a positive value." in str(response.content)
    if __name__ == "__main__":
        pytest.main()


@pytest.fixture()
def client():
    with TestClient(app) as tc:
        yield tc

        def test_process_wallet(client):
            response = client.post(
                "/wallet-score",
                json={"password_strength": 8, "two_factor_enabled": False},
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
            assert "wallet" in data.keys()
            assert "security_score" in data.keys()

            def test_invalid_wallet_security_score(client):
                response = client.post(
                    "/wallet-score",
                    json={"password_strength": 1, "two_factor_enabled": False},
                )
                assert response.status_code == 400


@pytest.fixture
def client():
    with TestClient(app) as _app:
        yield _app

        def test_verify_state(client):
            response = client.get("/verify-state")
            assert response.status_code == 405
            assert "Method Not Allowed" in str(response.content)

            def test_verify_state_valid_response(client):
                response = client.post(
                    "/verify-state",
                    content_type="application/json",
                    data=json.dumps({"result": "state verified successfully"}),
                )
                assert response.status_code == 200
                assert "state verified successfully" in str(response.content)

                def test_verify_state_missing_fields(client):
                    with pytest.raises(HTTPException) as exc:
                        client.post(
                            "/verify-state",
                            content_type="application/json",
                            data=json.dumps(
                                {"sender_chain": "ABC", "receiver_chain": "XYZ"}
                            ),
                        )
                        assert "Missing required fields" in str(exc.value)

                        def test_verify_state_invalid_response(client):
                            with pytest.raises(HTTPException) as exc:
                                client.post(
                                    "/verify-state",
                                    content_type="application/json",
                                    data=json.dumps({}),
                                )
                                assert "Method Not Allowed" in str(exc.value)


@pytest.fixture
def client():
    yield TestClient(app)

    def test_realtime_smart_contract_monitoring(client):
        response = client.get("/monitoring")
        assert response.status_code == 200
        data = json.loads(response.text)
        assert "events" in data
        assert len(data["events"]) > 0


@pytest.fixture()
def client():
    yield TestClient(app)

    def test_add_token_vesting(client):
        with pytest.raises(HTTPException) as context:
            client.get("/token_vestings")
            assert "Invalid lockup period" in str(context.value)

            def test_get_all_token_vestings(client):
                response = client.get("/token_vestings")
                all_vestings = response.json()
                assert len(all_vestings) > 0


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_get_total_yield(client):
            response = client.get("/total-yield/ETH")
            assert response.status_code == 200
            assert isinstance(response.json(), float)

            def test_get_total_yield_not_found(client):
                response = client.get("/total-yield/BSC")
                assert response.status_code == 404


@pytest.fixture
def client():
    client = TestClient(app)
    return client


async def test_optimize_liquidity(client):
    response = client.get("/optimize-liquidity")
    assert response.status_code == 405
    assert "Method Not Allowed" in str(response.content)

    def test_optimized_endpoint_returns_status_and_details(client):
        with pytest.raises(HTTPException):
            response = client.get("/endpoint")
            data = response.json()
            assert data["status"] == "optimized"
            assert isinstance(data["optimization_details"], list)


def test_withdraw_request():
    client = TestClient(app)
    data = {"amount": 500, "address": "123 Test St"}
    response = client.post("/withdraw", json=data)
    assert response.status_code == 200
    content = response.json()
    assert content["amount"] == 500
    assert content["balance"] > 1000
    data_insufficient_funds = {"amount": 600, "address": "123 Test St"}
    response_insufficient_funds = client.post("/withdraw", json=data_insufficient_funds)
    assert response_insufficient_funds.status_code == 400
    content_insufficient_funds = response_insufficient_funds.json()
    assert (
        content_insufficient_funds["detail"] == "Insufficient funds for the withdrawal."
    )


def test_bulk_approve_endpoint():
    client = TestClient(app)
    with pytest.raises(HTTPException):
        response = client.get("/bulk-approve")
        print(response.status_code)
        assert response.status_code == 200


def test_create_network():
    client = TestClient(app)
    response = client.post("/network", json={"network_id": "test-network", "nodes": []})
    assert response.status_code == 200
    created_network = response.json()
    assert created_network["network_id"] == "test-network"
    assert len(created_network["nodes"]) == 0

    def test_get_network():
        client = TestClient(app)
        response = client.get("/network/test-network")
        assert response.status_code == 200
        network = response.json()
        assert network["network_id"] == "test-network"
        assert len(network["nodes"]) == 0

        def test_invalid_nodes_create_network():
            client = TestClient(app)
            with pytest.raises(HTTPException):
                response = client.post(
                    "/network", json={"network_id": "test-network", "nodes": []}
                )
                assert response.status_code == 400


@pytest.fixture()
def client():
    with TestClient(app) as Client:
        yield Client

        def test_create_verification_request(client):
            response = client.post(
                "/verify", json={"user_identity": "123", "verifier_signature": "abc"}
            )
            assert response.status_code == 200

            def test_get_verification_status(client):
                response = client.get("/verify", params={"verification_id": "1"})
                assert response.status_code == 200


def test_rebalance_endpoint():
    client = TestClient(app)
    response = client.get("/rebalance")
    assert response.status_code == 200
    data = response.json()
    assert "rebalance_status" in data
    assert "last_rebalanced_time" in data

    @pytest.mark.asyncio
    async def test_rebalance_endpoint_async():
        client = TestClient(app)
        response = await client.get("/rebalance")
        assert response.status_code == 200
        data = response.json()
        assert "rebalance_status" in data
        assert "last_rebalanced_time" in data


@pytest.mark.parametrize(
    "token_id,new_weight,expected_response",
    [("abc123", 0.6, {"operation_id": "f3e4d56-7890-a1b2-c3d4-5678e9"})],
    arguments="token_id new_weight expected_response",
)
def test_rebalance_tokens(token_id, new_weight, expected_response):
    response = rebalance_tokens(token_id=token_id, new_weight=new_weight)
    assert response == expected_response


app = FastAPI()
accounts = [Account(id="1", name="Main Account", owner_id="user123")]


class Account(BaseModel):
    id: str
    name: str
    owner_id: str

    @staticmethod
    async def get_account(account_id: str):
        for account in accounts:
            if account.id == account_id:
                return account
            raise HTTPException(status_code=404, detail="Account not found")

            @pytest.fixture(autospec=True)
            def create_delegation(client):

                def _create_delegation(subaccount):
                    response = client.post("/delegations", json=subaccount)
                    assert response.status_code == 200
                    return response.json()

                return _create_delegation

            def test_create_sub_account(create_delegation, client):
                subaccount = SubAccount(
                    id=str(uuid.uuid4()),
                    parent_id=None,
                    name="Sub-Account",
                    owner_id="user123",
                )
                response = create_delegation(subaccount)
                assert response["id"] == str(uuid.uuid4())
                assert response["parent_id"] is None
                assert response["name"] == "Sub-Account"
                assert response["owner_id"] == "user123"

                def test_create_main_account(client):
                    main_account = Account(
                        id="1", name="Main Account", owner_id="user123"
                    )
                    response = client.get("/accounts/1")
                    assert response.status_code == 200
                    account = response.json()
                    assert account["id"] == "1"
                    assert account["name"] == "Main Account"
                    assert account["owner_id"] == "user123"


def test_create_did():
    dids_repository = DidsRepository()
    client = TestClient(app)
    response = client.post("/dids")
    assert response.status_code == 200
    assert "message" in response.json()

    def test_get_did():
        dids_repository = DidsRepository()
        client = TestClient(app)
        response = client.post("/dids")
        new_did_id = response.json()["did"]
        get_response = client.get(f"/dids/{new_did_id}")
        assert get_response.status_code == 200
        assert "verificationMethod" in get_response.json()
        unknown_did_id = uuid.uuid4().hex
        unknown_get_response = client.get(f"/dids/{unknown_did_id}")
        assert unknown_get_response.status_code == 404

        def test_create_duplicate_did():
            dids_repository = DidsRepository()
            response = client.post("/dids")
            assert "409" in str(response.content)
            with pytest.raises(HTTPException):
                _ = response.json()
                with pytest.raises(ValueError):
                    _ = DIDDocument(id=None)

                    def test_generate_did_document():
                        did_document = generate_did_document()
                        assert isinstance(did_document, DIDDocument)
                        assert did_document.id
                        assert did_document.verificationMethod
                        assert did_document.authentication


def setup_module(module):
    app.dependency_overrides = {
        app.get: lambda *_: [
            {
                "id": "test_position",
                "data": {"symbol": "AAPL", "quantity": 10, "price": 150.0},
            }
        ]
    }

    def test_get_position():
        client = TestClient(app)
        response = client.get("/position/test_position")
        assert response.status_code == 200
        position_data = response.json()
        assert position_data["symbol"] == "AAPL"
        assert position_data["quantity"] == 10
        assert position_data["price"] == 150.0

        def teardown_module(module):
            del app.dependency_overrides[app.get]


@pytest.mark.asyncio
async def test_update_oracle_prices():
    start = time.time()
    async with TestClient(app) as client:
        response = await client.get("/update_oracle_prices")
        assert response.status_code == 200
        result_data = response.json()
        end = time.time()
        elapsed_time = end - start
        assert "new_oracle_prices" in result_data
        assert isinstance(result_data["new_oracle_prices"], list)

        def test_endpoint():
            client = TestClient(app)
            response = client.get("/endpoint")
            assert response.status_code == 200


def test_create_transaction():
    app = FastAPI()

    @app.post("/transaction")
    async def create_transaction(transaction_data: Transaction):
        transaction.id = str(uuid.uuid4())
        return transaction

    client = TestClient(app)
    response = client.post(
        "/transaction",
        json={
            "id": str(uuid.uuid4()),
            "from_account": "Account1",
            "to_account": "Account2",
            "amount": 100.0,
        },
    )
    assert response.status_code == 200
    transaction = response.json()
    assert "id" in transaction

    def test_get_transaction():
        app = FastAPI()

        @app.post("/transaction")
        async def create_transaction(transaction_data: Transaction):
            transaction.id = str(uuid.uuid4())
            return transaction

        client = TestClient(app)
        response = client.post(
            "/transaction",
            json={
                "id": str(uuid.uuid4()),
                "from_account": "Account1",
                "to_account": "Account2",
                "amount": 100.0,
            },
        )
        assert response.status_code == 200
        transaction_data = response.json()
        response = client.get(
            "/transaction/{id}", params={"id": transaction_data["id"]}
        )
        assert response.status_code == 200
        fetched_transaction = response.json()
        assert fetched_transaction == transaction_data


@pytest.fixture
def client():
    with TestClient(app) as _app:
        yield _app

        def test_create_solvency_proof(client):
            response = client.post(
                "/solvency-proof",
                files={"file": "test_file"},
                json={"data": {"name": "John Doe", "dob": "2020-01-01"}},
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict) and "id" in data


def create_test_client():
    return TestClient(app)


@pytest.fixture
def test_kyc_document():
    yield KycDocument(type="SSN", content="123-45-6789")

    @pytest.fixture
    def test_user_kyc(test_kyc_document):
        user_kyc = UserKyc(name="John Doe", documents=[test_kyc_document])
        return user_kyc

    def test_create_user_kyc_valid():
        client = create_test_client()
        response = client.get("/create_user_kyc")
        assert response.status_code == 200
        created_user_kyc = response.json()
        assert isinstance(created_user_kyc, UserKyc)

        def test_create_user_kyc_invalid(test_user_kyc):
            with pytest.raises(HTTPException):
                create_user_kyc(user_kyc=test_user_kyc)


def test_add_collateral():
    client = TestClient(app)
    response = client.post(
        "/collaterals",
        content={"chain": "test_chain", "amount": 1.0},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    data = json.loads(response.text)
    assert isinstance(data, list)
    assert len(data) > 0
    assert isinstance(data[0], dict)
    assert "chain" in data[0]
    assert "amount" in data[0]

    def test_get_collaterals():
        client = TestClient(app)
        response_add = client.post(
            "/collaterals",
            content={"chain": "test_chain", "amount": 1.0},
            headers={"Content-Type": "application/json"},
        )
        assert response_add.status_code == 200
        with open("main/collaterals.json", "r") as f:
            collaterals = json.load(f)
            assert len(collaterals) > 0
            assert isinstance(collaterals[0], dict)

            def test_update_collateral():
                client = TestClient(app)
                response_add = client.post(
                    "/collaterals",
                    content={"chain": "test_chain", "amount": 1.0},
                    headers={"Content-Type": "application/json"},
                )
                assert response_add.status_code == 200
                with open("main/collaterals.json", "r") as f:
                    collaterals = json.load(f)
                    initial_collateral_data = {c["chain"]: c for c in collaterals}
                    response_update = client.put(
                        "/collaterals/test_chain",
                        content={"chain": "test_chain", "amount": 2.0},
                        headers={"Content-Type": "application/json"},
                    )
                    assert response_update.status_code == 200
                    with open("main/collaterals.json", "r") as f:
                        collaterals = json.load(f)
                        updated_collateral_data = {c["chain"]: c for c in collaterals}
                        assert initial_collateral_data["test_chain"]["amount"] == 1.0
                        assert updated_collateral_data["test_chain"]["amount"] == 2.0


def test_create_token_event():
    client = TestClient(app)
    token_event_data = {
        "event_type": "login",
        "user_id": str(uuid.uuid4()),
        "timestamp": datetime.now(),
    }
    response = client.post("/token-event", data=token_event_data)
    assert response.status_code == 200

    def test_read_token_events():
        client = TestClient(app)
        with open("test_token_events.json", "w") as f:
            json.dump(
                [
                    {
                        "id": "1",
                        "event_type": "login",
                        "user_id": "user1234",
                        "timestamp": "2022-08-01T10:00:00Z",
                    }
                ],
                f,
            )
            response = client.get("/token-events")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            event = data[0]
            assert event["id"] == "1"
            assert event["event_type"] == "login"
            assert event["user_id"] == "user1234"
            assert event["timestamp"] == "2022-08-01T10:00:00Z"

            def test_read_token_events_no_data():
                client = TestClient(app)
                response = client.get("/token-events")
                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, list)
                assert len(data) == 0


@pytest.fixture
def state_verifier():
    return StateVerifier()


def test_state_verifier_creation(state_verifier):
    assert isinstance(state_verifier, StateVerifier)

    def test_verify_state(state_verifier: StateVerifier, monkeypatch):
        requested_chain_id = "test"
        requested_state_root = "root"

        def mock_network_is_valid_chain(chain):
            if chain == requested_chain_id:
                return True
            return False

        monkeypatch.setattr("main.Network.is_valid_chain", mock_network_is_valid_chain)
        with pytest.raises(HTTPException):
            verify_response = state_verifier.verify_state(
                requested_chain_id, requested_state_root
            )
            assert isinstance(verify_response, dict)

            def test_verify_state_no_chain(state_verifier: StateVerifier, monkeypatch):
                requested_chain_id = "non_existent"

                def mock_network_is_valid_chain(chain):
                    return False

                monkeypatch.setattr(
                    "main.Network.is_valid_chain", mock_network_is_valid_chain
                )
                with pytest.raises(HTTPException):
                    verify_response = state_verifier.verify_state(
                        requested_chain_id, "root"
                    )

                    def test_verify_state_invalid_chain(
                        state_verifier: StateVerifier, monkeypatch
                    ):
                        requested_chain_id = "test"

                        def mock_network_is_valid_chain(chain):
                            return False

                        monkeypatch.setattr(
                            "main.Network.is_valid_chain", mock_network_is_valid_chain
                        )
                        with pytest.raises(HTTPException):
                            verify_response = state_verifier.verify_state(
                                requested_chain_id, "root"
                            )

                            def test_verify_state_success(
                                state_verifier: StateVerifier, monkeypatch
                            ):
                                requested_chain_id = "test"
                                requested_state_root = "root"

                                def mock_network_is_valid_chain(chain):
                                    return True

                                monkeypatch.setattr(
                                    "main.Network.is_valid_chain",
                                    mock_network_is_valid_chain,
                                )
                                verify_response = state_verifier.verify_state(
                                    requested_chain_id, requested_state_root
                                )
                                assert (
                                    isinstance(verify_response, dict)
                                    and "state_hash" in verify_response
                                )


def test_start_protocol():
    response = app.router.start_protocol(app.MPC())
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200

    def test_get_value_valid_party_id():
        value = generate_random_value()
        response = app.router.get_value(app.MPC(), "valid_party_id")
        assert isinstance(response, JSONResponse)
        assert response.status_code == 200
        assert response.content["value"] == value

        def test_get_value_invalid_party_id():
            with pytest.raises(HTTPException):
                _ = app.router.get_value(app.MPC(), "invalid_party_id")

                def setup_module(module):
                    module.app.router.include_router(app.MPC.as_router())


client = TestClient(app)


def test_get_oracles():
    response = client.get("/oracles")
    assert response.status_code == 200
    assert "oracles" in response.json()
    assert len(response.json()["oracles"]) > 0

    def test_get_oracle():
        response = client.get("/oracle/DefaultOracle")
        assert response.status_code == 200
        assert "oracle" in response.json()
        assert response.json()["oracle"].name == "DefaultOracle"
        assert response.json()["oracle"].address == "123 Main St, Anytown USA"


@pytest.fixture
def delegation_pool():
    DELEGATION_POOLS = {}

    def create_delegation_pool(pool_name, delegator_address, validator_address):
        if pool_name in DELEGATION_POOLS:
            raise HTTPException(status_code=400, detail="Pool already exists.")
            new_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            end_time = None
            delegation_pool = DelegationPool(
                id=new_id,
                pool_name=pool_name,
                delegator_address=delegator_address,
                validator_address=validator_address,
                start_time=start_time,
                end_time=end_time,
            )
            DELEGATION_POOLS[new_id] = delegation_pool
            return delegation_pool
        yield
        for id, pool in DELEGATION_POOLS.items():
            DELEGATION_POOLS[id].end_time = datetime.utcnow()
            DELEGATION_POOLS[id].save()

            def test_create_delegation_pool(delegation_pool):
                create_delegation_pool("pool1", "delegator1", "validator1")
                assert delegation_pool.pool_name == "pool1"
                assert delegation_pool.delegator_address == "delegator1"
                assert delegation_pool.validator_address == "validator1"

                def test_create_duplicate_delegation_pool():
                    with pytest.raises(HTTPException):
                        create_delegation_pool("pool1", "delegator2", "validator2")
                        create_delegation_pool("pool1", "delegator3", "validator3")

                        def test_get_delegation_pool():
                            delegation_pool = create_delegation_pool(
                                "pool1", "delegator1", "validator1"
                            )
                            get_delegation_pool(delegation_pool.id())
                            assert True


@pytest.fixture
def strategy():
    yield Strategy()

    def test_create_yield_strategy():
        response = app.test_client().post("/yield-strategy", json={})
        assert response.status_code == 200
        assert "strategy_id" in response.json()
        assert isinstance(response.json()["strategy_id"], str)


@pytest.fixture
def client():
    with TestClient(app) as FastAPIClient:
        yield FastAPIClient

        def test_create_execution(client):
            response = client.post(
                "/algorithmic-execution", json={"symbol": "AAPL", "quantity": 100}
            )
            assert response.status_code == 200
            execution_data = response.json()
            assert "id" in execution_data

            def test_get_execution(client):
                first_response = client.post(
                    "/algorithmic-execution", json={"symbol": "AAPL", "quantity": 100}
                )
                second_response = client.get(
                    f"/algorithmic-execution/{first_response.json()['id']}]"
                )
                assert second_response.status_code == 200
                execution_data = second_response.json()
                assert "id" in execution_data


@pytest.mark.parametrize("status", ["created", "cancelled"])
def test_cancel_order(client, status):
    order_id = str(uuid.uuid4())
    response = client.post("/orders", json={"items": [], "status": "created"})
    created_order = OrderRepository().get_order_by_id(response.json()["id"])
    response = client.put(f"/orders/{created_order.id}", json={"status": status})
    assert response.status_code == 200
    cancelled_order = OrderRepository().get_order_by_id(response.json()["id"])
    assert cancelled_order["status"] == status

    def test_cancel_order_not_found(client):
        order_id = str(uuid.uuid4())
        try:
            response = client.put(f"/orders/{order_id}", json={"status": "cancelled"})
            assert False, f"Expected HTTP 404 but got HTTP {response.status_code}"
        except HTTPException as e:
            pass


importpytest


def test_kyc_endpoint():
    response = app.test_client().get("/kyc")
    assert response.status_code == 405

    def test_valid_document_file():
        test_client = TestClient(app)

        def mock_validate_document(document):
            return "Valid Document"

        with pytest.raises(Exception):
            with mock.patch(
                "main.validate_document", side_effect=mock_validate_document
            ):
                response = test_client.post(
                    "/kyc", files={"file": open("valid_document.pdf", "rb")}
                )
                assert b"KYC verification successful" in response.content
                assert response.status_code == 200

                def test_invalid_document_file():
                    test_client = TestClient(app)

                    def mock_validate_document(document):
                        raise Exception("Invalid Document")
                        with pytest.raises(Exception):
                            with mock.patch(
                                "main.validate_document",
                                side_effect=mock_validate_document,
                            ):
                                response = test_client.post(
                                    "/kyc",
                                    files={"file": open("invalid_document.pdf", "rb")},
                                )
                                assert b"KYC verification failed" in response.content
                                assert response.status_code == 400


@pytest.fixture()
def client():
    yield TestClient(app)

    def test_get_markets_data(client):
        response = client.get("/markets")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert all((isinstance(item, dict) for item in response.json()))

        def test_execute_arbitrage(client):
            with pytest.raises(HTTPException):
                data = {"symbol": "AAPL", "exchange": "NYSE"}
                response = client.post("/execute_arbitrage", json=data)
                assert response.status_code == 500


app = FastAPI()


def test_create_reward_snapshot():
    snapshot = create_reward_snapshot(total_staked=100, reward_percentage=5)
    assert snapshot.total_staked == 100
    assert snapshot.reward_percentage == 5

    def test_get_reward_snapshot_unexpected_exception():
        with pytest.raises(HTTPException):
            get_reward_snapshot()

            @pytest.mark.asyncio
            async def test_get_rewards():
                client = TestClient(app)
                response = await client.get("/snapshot")
                data = response.json()
                assert data["timestamp"].replace(
                    microsecond=0
                ) != datetime.utcnow().replace(microsecond=0)


def test_margin_transfer_endpoint():
    with pytest.raises(HTTPException):
        response = app.test_client().post("/transfer/")
        assert response.status_code == 400
        data = {"sender_account_id": "sender1", "receiver_account_id": "receiver2"}
        margin_transfer = MarginTransfer(
            sender_account_id="sender1", receiver_account_id="receiver2"
        )
        response = app.test_client().post("/transfer/", json=data)
        assert response.status_code == 200
        expected_response = {
            "message": "Margin transfer request processed successfully",
            "margin_transfer": margin_transfer.__dict__,
        }
        assert response.json() == expected_response

        def test_margin_transfer_validation():
            data = {"sender_account_id": "invalid", "receiver_account_id": "valid"}
            with pytest.raises(HTTPException):
                MarginTransfer(sender_account_id="invalid", receiver_account_id="valid")


@pytest.fixture
def payment_channel():
    return PaymentChannel(id=1, network_id="NetworkA", last_update=datetime.now())


def test_create_payment_channel(payment_channel):
    new_payment_channel = create_payment_channel(payment_channel)
    assert new_payment_channel.id == 1
    assert new_payment_channel.network_id == "NetworkA"
    assert new_payment_channel.last_update == datetime.now()

    def test_get_payment_channel():
        pass

        def test_update_payment_channel(payment_channel):
            updated_payment_channel = PaymentChannel(
                id=1, network_id="NetworkB", last_update=datetime.now()
            )
            response = update_payment_channel(1, updated_payment_channel)
            assert response.id == 1
            assert response.network_id == "NetworkB"
            assert response.last_update == datetime.now()


def test_create_compliance_report():
    with pytest.raises(HTTPException):
        response = TestClient(app).post(
            "/compliance-report",
            json={"report_id": 123, "date_created": datetime.now()},
        )
        assert response.status_code == 400

        @patch("main.datetime")
        def test_get_current_date(mock_datetime):
            mock_datetime.now.return_value = datetime(2023, 1, 5, 10, 30)
            with pytest.raises(HTTPException):
                response = TestClient(app).get("/current-date")
                assert response.status_code == 200


def create_app():
    return app


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c

        def test_generate_proof_of_reserves(client):
            response = client.post(
                "/proof-of-reserves",
                json={
                    "stablecoins": {"USD": 1000, "EUR": 500},
                    "reserves": {"USD": 2000, "EUR": 1500},
                },
            )
            assert response.status_code == 400
            data = response.json()
            assert (
                "Proof of Reserves generation period is from day 1 to day 14."
                == data["detail"]
            )

            def test_generate_proof_of_reserves_invalid_date():
                client = TestClient(create_app())
                with pytest.raises(HTTPException):
                    response = client.get("/proof-of-reserves")
                    assert response.status_code == 400
                    data = response.json()
                    assert "Not a valid date" == data["detail"]


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_valid_deposit(client):
            data = {
                "type": "deposit",
                "id": "1234567890",
                "amount": 100.0,
                "timestamp": datetime.now(),
            }
            response = client.post("/batch/", json=data)
            assert response.status_code == 200
            result = response.json()
            assert (
                "message" in result
                and result["message"]
                == "Batch with ID '1234567890' processed successfully."
            )

            def test_valid_withdrawal(client):
                data = {
                    "type": "withdrawal",
                    "id": "0987654321",
                    "amount": 50.0,
                    "timestamp": datetime.now(),
                }
                response = client.post("/batch/", json=data)
                assert response.status_code == 200
                result = response.json()
                assert (
                    "message" in result
                    and result["message"]
                    == "Batch with ID '0987654321' processed successfully."
                )

                def test_invalid_type(client):
                    data = {
                        "type": "invalidType",
                        "id": "0123456789",
                        "amount": 30.0,
                        "timestamp": datetime.now(),
                    }
                    response = client.post("/batch/", json=data)
                    assert response.status_code == 400
                    result = response.json()
                    assert (
                        "detail" in result
                        and result["detail"]
                        == "Invalid type. Must be 'deposit' or 'withdrawal'."
                    )

                    def test_empty_request(client):
                        data = {}
                        response = client.post("/batch/", json=data)
                        assert response.status_code == 400
                        result = response.json()
                        assert (
                            "description" in result
                            and result["description"]
                            == "invalid request: Empty JSON body."
                        )


def test_get_concentrated_liquidity():
    client = TestClient(app)
    expected_data = {"concentration": 0.5}
    response = client.get("/concentrated_liquidity")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["concentration"] == expected_data["concentration"]
    print("Test passed.")


@pytest.main
def test_create_transaction(client: TestClient):
    response = client.post(
        "/transaction", json={"amount": 1000, "currency": "USD", "type": "deposit"}
    )
    assert response.status_code == 200
    assert len(response.json()["data"].transactions) == 1

    @pytest.main
    def test_get_transactions(client: TestClient):
        response = client.get("/transactions")
        assert response.status_code == 200
        assert len(response.json()["data"]) == 5

        @pytest.main
        def test_get_transaction_not_found(client: TestClient):
            response = client.get("/transactions/12345")
            assert response.status_code == 404
            assert response.json() == {"detail": "Transaction not found."}

            @pytest.main
            def test_get_transaction(client: TestClient):
                response = client.get("/transactions/1")
                assert response.status_code == 200
                assert len(response.json()["data"].transactions) == 1


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (
            {
                "start_time": "2022-01-01",
                "end_time": "2022-02-01",
                "target_gas_price": 0.1,
            },
            "Implement your gas optimization strategy here...",
        )
    ],
)
def test_gas_optimization(input_data, expected_output):
    client = TestClient(app)
    response = client.post("/gas_optimization", json=input_data)
    assert response.status_code == 200
    result = response.json()
    assert "gas_optimization" in result.keys()
    assert result["gas_optimization"] == expected_output


def test_event_post_request(test_app: TestClient):
    event_data = {
        "event_type": "token-distribution",
        "token_id": 1,
        "recipient_address": "0x123...",
        "timestamp": datetime.now(),
    }
    response = test_app.post("/events", json=event_data)
    assert response.status_code == 200


def test_migration_params():
    migration_params = MigrationParams(
        source_pool_id="source_pool_id", target_pool_id="target_pool_id", amount=100
    )
    assert migration_params.source_pool_id == "source_pool_id"
    assert migration_params.target_pool_id == "target_pool_id"
    assert migration_params.amount == 100

    def test_migration_success():
        response = migrate_pool(
            MigrationParams(
                source_pool_id="source_pool", target_pool_id="target_pool", amount=10
            )
        )
        expected_response = {
            "message": "Pool migration successful!",
            "migration_id": "a2b3c4d-e5f6-7-8-9",
        }
        assert isinstance(response, dict)
        assert response == expected_response

        def test_migration_failure():
            with pytest.raises(HTTPException):
                migrate_pool(
                    MigrationParams(
                        source_pool_id="source_pool",
                        target_pool_id="target_pool",
                        amount=0,
                    )
                )


def test_get_oracle_exchange_rates():
    client = TestClient(app)
    response = client.get("/oracles")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert all((isinstance(item, OracleData) for item in response.json()))
    if __name__ == "__main__":
        pytest.main()


def setup_module(module):
    client = TestClient(app)
    app.dependency_overrides[app.delegation_pool.get_all_delegation_pools] = (
        lambda: [
            {
                "id": str(uuid.uuid4()),
                "pool_name": "Test Delegation Pool",
                "delegator_address": "test_delegator",
                "validator_address": "test_validator",
            }
        ],
    )
    client = None
    setup_test_data = True

    @pytest.mark.parametrize(
        "input_values, expected_output",
        [
            (
                {
                    "id": "existing_pool_id",
                    "pool_name": "Test Pool Update",
                    "delegator_address": "updated_delegator",
                    "validator_address": "updated_validator",
                },
                None,
            ),
            (
                {
                    "pool_name": "Test Pool Update",
                    "delegator_address": "updated_delegator",
                    "validator_address": "updated_validator",
                },
                "existing_pool_id",
            ),
        ],
    )
    def test_create_update_delete_delegation_pools(client):
        test_data = {"input_values": {}, "expected_output": None}
        if setup_test_data:
            input_value = test_data["input_values"]
            expected_output = test_data["expected_output"]
            response = client.post("/create_new_delegation_pool", json=input_value)
            pool_data = response.json()
            assert pool_data is not None
            assert "id" in pool_data
            input_values_update = {
                "pool_name": "Updated Test Delegation Pool",
                "delegator_address": "updated_test_delegator",
                "validator_address": "updated_test_validator",
            }
            response = client.put(
                f"/update_delegation_pool/{pool_data['id']}", json=input_values_update
            )
            assert response.status_code == 200
            pool_data_updated = response.json()
            assert "id" in pool_data_updated
            response = client.delete(f"/delete_delegation_pool/{pool_data['id']}")
            assert response.status_code == 200
        else:
            pytest.skip("Test data setup failed. Skipping test.")

            @pytest.mark.parametrize(
                "endpoint, method, expected_status_code",
                [
                    ("create_new_delegation_pool", "POST", 200),
                    ("update_delegation_pool/{pool_id}", "PUT", 200),
                    ("delete_delegation_pool/{pool_id}", "DELETE", 200),
                ],
            )
            def test_valid_endpoints(client):
                response = client.request("GET", endpoint)
                assert response.status_code == expected_status_code

                @pytest.mark.parametrize(
                    "input_values, expected_output",
                    [
                        (
                            {
                                "delegator_address": "test_delegator",
                                "validator_address": "test_validator",
                            },
                            200,
                        )
                    ],
                )
                def test_invalid_endpoints(client):
                    input_value = {"delegator_address": "invalid_test_delegator"}
                    response = client.post(
                        "/create_new_delegation_pool", json=input_value
                    )
                    assert response.status_code == expected_output


def test_monitor_smart_contract():
    response = app.test_client().get("/monitor")
    assert response.status_code == 200

    def test_monitor_smart_contract_state_change():
        response = app.test_client().get("/monitor")
        assert "ACTIVE" in str(response.text)


def test_create_derivative():
    settlement_manager = SettlementManager()
    derivative = settlement_manager.create_derivative(datetime(2023, 1, 1), 1000.0)
    assert isinstance(derivative, Derivative)
    assert derivative.contract_id is not None
    assert derivative.settlement_date is None
    assert derivative.settlement_amount == 1000.0

    def test_settle_derivative():
        settlement_manager = SettlementManager()
        settlement_manager.create_derivative(datetime(2023, 1, 1), 1000.0)
        with pytest.raises(HTTPException):
            settlement_manager.settle_derivative("existing_contract_id")
            derivative = settlement_manager.get_derivative_by_id("new_contract_id")
            assert derivative.contract_id == "new_contract_id"
            assert derivative.settlement_date is not None
            assert derivative.settlement_amount > 0


@pytest.mark.parametrize(
    "input, expected_status_code, expected_error_message",
    [
        (
            {
                "id": "1",
                "name": "John Doe",
                "description": "Test oracle",
                "address": "123 Test Street",
            },
            200,
            "Oracle created successfully",
        ),
        (
            {"id": "1"},
            400,
            "Invalid data provided. Please ensure the 'name', 'description' and 'address' fields are all filled with valid values.",
        ),
    ],
)
def test_create_reputation_oracle(
    input_data, expected_status_code, expected_error_message
):
    client = TestClient(app)
    response = client.post("/oracle", json=input_data)
    assert response.status_code == expected_status_code
    if expected_status_code == 400:
        assert "Invalid data provided" in str(response.content)
    else:
        assert "Oracle created successfully" in str(response.content)

        @pytest.mark.parametrize(
            "input, expected_status_code, expected_error_message",
            [
                ({"id": "1"}, 200, "Oracle retrieved successfully"),
                ({"id": "nonexistent_id"}, 404, "Oracle not found with the given ID"),
            ],
        )
        def test_get_reputation_oracle(
            input_data, expected_status_code, expected_error_message
        ):
            client = TestClient(app)
            response = client.get(f"/oracle/{input_data['id']}")
            assert response.status_code == expected_status_code
            if expected_status_code == 404:
                assert "Oracle not found with the given ID" in str(response.content)
            else:
                assert "Oracle retrieved successfully" in str(response.content)

                @pytest.mark.parametrize(
                    "input, expected_status_code, expected_error_message",
                    [
                        (
                            {
                                "id": "1",
                                "name": "Test Updated Oracle",
                                "description": "Updated Oracle Test",
                                "address": "456 New Street",
                            },
                            200,
                            "Oracle updated successfully",
                        )
                    ],
                )
                def test_update_reputation_oracle(
                    input_data, expected_status_code, expected_error_message
                ):
                    client = TestClient(app)
                    response = client.put(f"/oracle/{input_data['id']}")
                    assert response.status_code == expected_status_code
                    assert "Oracle updated successfully" in str(response.content)

                    @pytest.mark.parametrize(
                        "input, expected_status_code, expected_error_message",
                        [({"id": "1"}, 200, "Oracle deleted successfully")],
                    )
                    def test_delete_reputation_oracle(
                        input_data, expected_status_code, expected_error_message
                    ):
                        client = TestClient(app)
                        response = client.delete(f"/oracle/{input_data['id']}")
                        assert response.status_code == expected_status_code
                        assert "Oracle deleted successfully" in str(response.content)


def test_create_position():
    client = TestClient(app)
    response = client.post(
        "/positions",
        content_type="application/json",
        data=json.dumps(
            {"chain": "TestChain", "token_address": "0xTokenAddress", "value": 10.0}
        ),
    )
    assert response.status_code == 200

    def test_get_positions():
        client = TestClient(app)
        response = client.get("/positions?chain=TestChain&token_address=0xTokenAddress")
        assert response.status_code == 200


@pytest.mark.parametrize("status_code", [200])
def test_cancel_order_endpoint_status_code(status_code):
    client = TestClient(app)
    response = client.get("/cancel-order/{order_id}", params={"order_id": "12345"})
    assert response.status_code == status_code

    def test_cancel_order_endpoint_invalid_order():
        client = TestClient(app)
        response = client.get("/cancel-order/{order_id}", params={"order_id": "99999"})
        assert response.status_code == 404
        assert response.json() == {"detail": "Order not found"}

        def test_cancel_order_endpoint_success():
            client = TestClient(app)
            response = client.get(
                "/cancel-order/{order_id}", params={"order_id": "12345"}
            )
            assert response.status_code == 200
            assert (
                response.json().get("detail")
                == "Order with ID 12345 has been canceled on [date and time]"
            )


@pytest.fixture
def client():
    yield TestClient(app)

    def test_upgrade_version(client):
        response = client.put("/db/migrate")
        assert response.status_code == 200

        def test_downgrade_version(client):
            response = client.delete("/db/migrate")
            assert response.status_code == 200

            def test_get_db_engine():
                engine = get_db_engine()
                assert isinstance(engine, create_engine)


@pytest.fixture(autosuppress=True)
def client():
    with TestClient(app) as tc:
        yield tc

        def test_create_audit_log():
            audit_log = AuditLog(
                timestamp=datetime.now(),
                action="created",
                user_id=1,
                resource="audit-log",
            )
            response = app.post("/audit-log/", json=audit_log.dict())
            assert response.status_code == 200
            assert "log_id" in response.json()

            def test_create_audit_log_empty_details():
                audit_log = AuditLog(
                    timestamp=datetime.now(),
                    action="created",
                    user_id=1,
                    resource="audit-log",
                    details="",
                )
                with pytest.raises(HTTPException):
                    app.post("/audit-log/", json=audit_log.dict())


def test_init():
    signers = ["Alice", "Bob"]
    ws = MultiSignatureWallet(signers)
    assert ws.signers == signers
    assert ws.approved is False
    assert ws.disapproved is False

    def test_approve():
        wallet = MultiSignatureWallet(["Alice", "Bob"])
        with pytest.raises(HTTPException):
            approve(wallet, [])
            signature = ["Alice"]
            with pytest.raises(HTTPException):
                approve(wallet, signature)
                signature = ["Alice", "Bob"]
                wallet.approved = False
                assert approve(wallet, signature)["result"] == "wallet approved"

                def test_disapprove():
                    wallet = MultiSignatureWallet(["Alice", "Bob"])
                    wallet.disapproved = False
                    assert disapprove(wallet) is None
                    wallet.disapproved = True
                    assert disapprove(wallet)["result"] == "wallet disapproved"


def test_derivative_stress_test_endpoint():
    client = TestClient(app)
    with pytest.raises(HTTPException):
        response = client.get("/stress-test")
        assert response.status_code == 200
        stress_data = response.json()
        assert "data" in stress_data
        data = stress_data["data"]
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

        def test_derivative_stress_test_results():
            with pytest.raises(NotImplementedError):
                stress_test_instance = StressTestData(data=None)
                stress_data = stress_test_instance.data


@pytest.fixture
def client():
    with TestClient(app) as ac:
        yield ac

        def test_get_desk_allocation(client):
            response = client.get("/allocation")
            assert response.status_code == 200
            assert "desk" in response.json()
            assert "allocation" in response.json()

            def test_get_desk_allocation_with_invalid_desk(client):
                with pytest.raises(HTTPException) as e:
                    response = client.get(f"/allocation?desk=invalid")
                    print(response.text)
                    assert "Desk not found" in str(e.value)
                    assert response.status_code == 404


def test_market_maker_endpoint():
    client = TestClient(app)
    response = client.get("/market_maker")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

    def test_market_maker():

        @app.get("/market_maker")
        def market_maker():
            if not MarketMaker.market_maker:
                market_data = {
                    "AAPL": {"last_price": 145.0, "volatility": 15},
                    "GOOG": {"last_price": 2700.0, "volatility": 8},
                }
                mm = MarketMaker(app)
                mm.initialize_market_maker(market_data)
                return {"market_maker": "initialized"}
            market_data = {
                "AAPL": {
                    "last_price": random.uniform(130, 150),
                    "volatility": random.randint(10, 20),
                },
                "GOOG": {
                    "last_price": random.uniform(2600, 2800),
                    "volatility": random.randint(5, 15),
                },
            }
            mm = MarketMaker(app)
            response = mm.initialize_market_maker(market_data)
            assert isinstance(response, Dict) and "market_maker" in response


@pytest.fixture
def client():
    with TestClient(app) as TC:
        yield TC

        def test_create_collateral(client):
            response = client.post(
                "/collaterals", json={"chain": "TestChain", "amount": 100.0}
            )
            assert response.status_code == 200
            data = response.json()
            assert "message" in data and data["message"] == "Collateral created"
            assert "data" in data and isinstance(data["data"], Collateral)
            assert data["data"].chain == "TestChain"
            assert data["data"].amount == 100.0

            def test_get_collaterals(client):
                response = client.get("/collaterals")
                assert response.status_code == 200
                data = response.json()
                assert (
                    "message" in data
                    and data["message"] == "Collaterals retrieved successfully"
                )
                assert "data" in data and isinstance(data["data"], list)
                assert len(data["data"]) > 0

                def test_update_collateral(client):
                    response = client.put(
                        "/collaterals/{id}",
                        json={"chain": "TestChain", "amount": 200.0},
                        params={"id": "1"},
                    )
                    assert response.status_code == 200
                    data = response.json()
                    assert (
                        "message" in data
                        and data["message"] == "Collateral updated successfully"
                    )
                    assert "data" in data and isinstance(data["data"], dict)
                    assert data["data"]["chain"] == "TestChain"
                    assert data["data"]["amount"] == 200.0

                    def test_delete_collateral(client):
                        response = client.delete(
                            "/collaterals/{id}", params={"id": "1"}
                        )
                        assert response.status_code == 200
                        data = response.json()
                        assert (
                            "message" in data
                            and data["message"] == "Collateral deleted successfully"
                        )


@pytest.fixture()
def validator_node_service():
    service = ValidatorNodeService()
    service.create_validator_node("Test Node", "123 Test St.", "testPublicKey")
    return service


def test_create_validator_node(validator_node_service):
    node = validator_node_service.get_validator_node("testId")
    assert node is not None
    assert node.name == "Test Node"
    assert node.state == "offline"

    def test_update_validator_node(validator_node_service):
        validator_node = validator_node_service.create_validator_node(
            name="Test Node", address="123 Test St.", public_key="testPublicKey"
        )
        validator_node.service.update_validator_node(
            node_id=validator_node.id, state="online"
        )
        node = validator_node_service.get_validator_node(validator_node.id)
        assert node is not None
        assert node.state == "online"

        def test_get_validator_node(validator_node_service):
            validator_node = validator_node_service.create_validator_node(
                name="Test Node", address="123 Test St.", public_key="testPublicKey"
            )
            node = validator_node_service.get_validator_node(node.id)
            assert node is not None
            assert node.name == "Test Node"


app = TestClient(Main.app)


def test_get_liquidity():
    response = app.get("/liquidity")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

    def test_get_liquidity_by_asset():
        for i in [1, 2]:
            with pytest.raises(HTTPException):
                assert app.get(f"/liquidity/{i}")
                response = app.get("/liquidity/1")
                assert response.status_code == 200
                data = response.json()
                assert isinstance(data, dict)


importpytest


def test_endpoint():
    endpoint = PositionRiskEndpoint()
    decomposition_data = PositionRiskDecomposition(
        market_value=100,
        weighted_exposure=[10, 20, 30],
        sector_exposure=[0.5, 0.3, 0.2],
        country_exposure=[0.1, 0.25, 0.6],
    )
    with pytest.raises(HTTPException) as e:
        decomposition_data = endpoint.get_position_risk_decomposition(
            decomposition_data
        )
        assert str(e.value) == "Data does not match the provided data structure."


app = your_app_instance


def test_create_delegation_pool():
    pool_id = uuid.uuid4()
    delegation_pool = DelegationPool(pool_id=pool_id)
    with patch("main.DelegationPool.router.post") as mock_post:
        result = app.create_delegation_pool(delegation_pool)
        assert result == delegation_pool
        mock_post.assert_called_once_with(
            data=delegation_pool.dict(), content_type="application/json"
        )

        @pytest.mark.parametrize(
            "invalid_field, error_message",
            [
                ("pool_id", "Pool ID cannot be empty"),
                ("delegator_address", "Delegator address cannot be empty"),
            ],
        )
        def test_get_delegation_pool(invalid_field, error_message):
            pool_id = uuid.uuid4()
            with pytest.raises(HTTPException) as ex:
                response = app.get_delegation_pool(pool_id)
                assert str(ex.value) == f"{error_message}"

                @pytest.mark.parametrize(
                    "pool_weight_value, expected_error",
                    [
                        (0, "Pool weight cannot be zero"),
                        (-1, "Pool weight must be a positive integer"),
                    ],
                )
                def test_update_delegation_pool(pool_weight_value, expected_error):
                    pool_id = uuid.uuid4()
                    delegation_pool = DelegationPool(
                        pool_id=pool_id, pool_weight=pool_weight_value
                    )
                    with pytest.raises(HTTPException) as ex:
                        updated_delegation_pool = app.update_delegation_pool(
                            pool_id=delegation_pool.pool_id,
                            delegation_pool=delegation_pool,
                        )
                        assert str(ex.value) == f"{expected_error}"

                        @pytest.mark.parametrize(
                            "invalid_field, error_message",
                            [
                                ("pool_id", "Pool ID cannot be empty"),
                                (
                                    "delegator_address",
                                    "Delegator address cannot be empty",
                                ),
                            ],
                        )
                        def test_delete_delegation_pool(invalid_field, error_message):
                            pool_id = uuid.uuid4()
                            with pytest.raises(HTTPException) as ex:
                                response = app.delete_delegation_pool(pool_id)
                                assert str(ex.value) == f"{error_message}"


async def test_volatility_surface_endpoint():
    client = TestClient(app)
    response = client.get("/volatility_surface")
    assert response.status_code == 200
    assert "message" in response.json()

    async def test_calculated_volatility_surface_endpoint():
        with raises(HTTPException):
            asyncio.run(test_volatility_surface_endpoint())
            assert (
                False
            ), "This is a placeholder assertion to ensure the function is called."

            async def test_error_in_calculating_volatility_surface_endpoint():
                with raises(Exception) as context:
                    asyncio.run(test_calculated_volatility_surface_endpoint())
                    assert str(context.exception) == "An error occurred:"


@pytest.fixture
def client():
    with TestClient(app) as SC:
        yield SC

        def test_get_institutional_prime_brokerage_data(client):
            response = client.get("/institutions/1")
            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert "name" in data
            assert "account_number" in data
            assert "total_funds" in data
            assert "risk_level" in data

            def test_get_institutional_prime_brokerage_data_not_found(client):
                response = client.get("/institutions/999")
                assert response.status_code == 404


def test_get_loans_valid_loan_id():
    client = TestClient(app)
    response = client.get("/loans/test_loan")
    assert response.status_code == 200
    assert "loan_id" in response.json()
    assert "user_id" in response.json()
    assert "collateral_type" in response.json()
    assert "amount" in response.json()
    assert "interest_rate" in response.json()
    assert "start_date" in response.json()
    assert "end_date" not in response.json()

    def test_get_loans_invalid_loan_id():
        client = TestClient(app)
        response = client.get("/loans/invalid_loan")
        assert response.status_code == 400
        assert "Invalid loan_id" in str(response.content)

        def test_create_loan_valid_input():
            client = TestClient(app)
            new_record = MarginLendingRecord(
                loan_id="test_loan",
                user_id="user_123",
                collateral_type="Equity",
                amount=1000000,
                interest_rate=5.0,
                start_date=datetime.now(),
                end_date=None,
            )
            response = client.post("/loan", json=new_record.dict())
            assert response.status_code == 200
            assert "loan_id" in response.json()
            assert isinstance(response.json()["loan_id"], str)
            assert "user_id" in response.json()
            assert "collateral_type" in response.json()
            assert "amount" in response.json()
            assert "interest_rate" in response.json()
            assert "start_date" in response.json()
            assert "end_date" not in response.json()

            def test_update_loan_valid_input():
                client = TestClient(app)
                new_record = MarginLendingRecord(
                    loan_id="test_loan",
                    user_id="user_123",
                    collateral_type="Equity",
                    amount=2000000,
                    interest_rate=7.5,
                    start_date=datetime.now(),
                    end_date=None,
                )
                response = client.put("/loans/test_loan", json=new_record.dict())
                assert response.status_code == 200
                assert "loan_id" in response.json()
                assert isinstance(response.json()["loan_id"], str)
                assert "user_id" in response.json()
                assert "collateral_type" in response.json()
                assert "amount" in response.json()
                assert "interest_rate" in response.json()
                assert "start_date" in response.json()
                assert "end_date" not in response.json()

                def test_delete_loan_valid_input():
                    client = TestClient(app)
                    response = client.delete("/loans/test_loan")
                    assert response.status_code == 200
                    assert "loan_id" in str(response.content)
                    assert isinstance(str(response.content)["loan_id"], str)


def test_create_uuid():
    assert isinstance(uuid.uuid4(), uuid.UUID)

    @mark.parametrize(
        "timestamp",
        [datetime(2023, 1, 1).timestamp(), datetime(2100, 1, 1).timestamp()],
    )
    def test_timestamp_to_datetime(timestamp):
        dt = datetime.fromtimestamps(timestamp)
        assert isinstance(dt, datetime), "Timestamp to DateTime conversion failed."

        @pytest.mark.asyncio
        async def test_optimize_spread():
            dummy_data = {
                "spread": 10,
                "min_price": 1000,
                "max_price": 1500,
                "current_price": 1200,
            }
            client = TestClient(app)
            response = await client.post("/optimize", json=dummy_data)
            assert response.status_code == 200
            expected_result = {
                "result": "Optimized spread data: {'spread': 10, 'min_price': 1000, 'max_price': 1500, 'current_price': 1200}'}"
            }
            result_json = response.json()
            assert result_json["result"] == expected_result["result"]


@pytest.fixture
def client():
    with TestClient(app) as ac:
        yield ac

        def test_withdraw_success(client):
            response = client.post(
                "/withdraw", json={"destination_address": "1", "amount": 200.0}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["result"] == "Withdrawal successful"
            assert data["new_balance"] >= 800.0

            def test_withdraw_insufficient_balance(client):
                response = client.post(
                    "/withdraw", json={"destination_address": "1", "amount": 10000.0}
                )
                assert response.status_code == 400
                assert response.json() == {
                    "detail": "Insufficient balance for withdrawal"
                }

                def test_withdraw_invalid_destination(client):
                    with pytest.raises(HTTPException) as e:
                        response = client.post(
                            "/withdraw",
                            json={"destination_address": "abc", "amount": 500.0},
                        )
                        assert response.status_code == 400
                        assert response.json() == {
                            "detail": "Invalid destination address provided."
                        }


app = None


@pytest.fixture
def client():
    global app
    with TestClient(app) as _app:
        app = _app
        yield _app
        app = None

        def test_kyc_verification_endpoint(client):
            response = client.post(
                "/kyc/verify", files={"documents": "tests/data/test_documents.jpg"}
            )
            assert response.status_code == 200
            content = response.json()
            assert "status" in content and content["status"] == "success"
            assert (
                "message" in content
                and content["message"] == "Documents received successfully."
            )
            assert "user_id" in content
            assert "documents" in content

            def test_kyc_documented_endpoint(client):
                user_id = "1234"
                response = client.get(f"/kyc/documented/{user_id}")
                assert response.status_code == 200
                content = response.json()
                assert "status" in content and content["status"] == "success"
                assert (
                    "message" in content
                    and content["message"] == "User KYC documents documented."
                )


def test_calculate_score():
    score = 1000
    assert TradingCompetition().calculate_score() == score

    @pytest.mark.asyncio
    async def test_get_leaderboard():
        client = TestClient(app)
        response = await client.get("/leaderboard")
        data = response.json()
        assert len(data["trades"]) > 10
        assert "username" in data and isinstance(data["username"], str)
        assert "score" in data

        @pytest.mark.asyncio
        async def test_update_leaderboard():
            client = TestClient(app)
            response = await client.post(
                "/leaderboard", json={"username": "NewUser", "score": 10}
            )
            leaderboard_data = response.json()
            assert leaderboard_data["username"] == "NewUser"
            assert leaderboard_data["score"] == 10


def test_wash_trading_detector():
    trades = [
        Trade(
            timestamp=datetime(2023, 1, 1),
            instrument="BTCUSDT",
            side="buy",
            quantity=0.2,
        ),
        Trade(
            timestamp=datetime(2023, 1, 1),
            instrument="ETHUSDT",
            side="sell",
            quantity=0.15,
        ),
    ]
    detector = Trade.WashTradingDetector()
    with pytest.raises(ValueError):
        detector.detect_wash_trading(trades)
        assert len(detector.trade_counts) == 2

        def test_detect_wash_trading_no_instruments():
            trades = []
            detector = Trade.WashTradingDetector()
            result = detector.detect_wash_trading(trades)
            expected_result = ValueError(
                "No instruments detected that exhibit wash trading patterns"
            )
            assert isinstance(result, expected_result)


def test_calculate_fee_valid_input():
    calculate_fee = MagicMock()
    congestion_level = 0.5
    base_fee = 10
    increment_factor = 0.05
    result = calculate_fee(
        congestion_level=congestion_level,
        base_fee=base_fee,
        increment_factor=increment_factor,
    )
    assert result == max(base_fee + base_fee * congestion_level * increment_factor, 1)

    def test_calculate_fee_invalid_congestion_level():
        with pytest.raises(HTTPException):
            calculate_fee(congestion_level=-0.5, base_fee=10, increment_factor=0.05)

            def test_network_congestion_valid_input():

                @patch("main.network_congestion")
                def test_network_congestion(self, mock_network_congestion):
                    expected = NetworkCongestion()
                    expected.congestion_level = 0.5
                    mock_network_congestion.return_value = expected
                    result = network_congestion()
                    assert isinstance(result, NetworkConggestion)
                    mock_network_congestion.assert_called_once()


@pytest.fixture
def client():
    with TestClient(app) as ac:
        yield ac

        def test_risk_factors(client):
            response = client.get("/risk_factors")
            assert response.status_code == 200
            data = response.json()
            assert "age" in data
            assert "gender" in data
            assert "cholesterol" in data
            assert "glucose" in data


@pytest.mark.asyncio
async def test_manipulation_detection_endpoint():
    client = TestClient(app)
    response = client.get("/manipulation-detection")
    assert response.status_code == 200
    assert "data" in response.json()


def test_list_cross_chain_collaterals():
    client = TestClient(app)
    response = client.get("/chains/{chain_id}/collaterals")
    assert response.status_code == 200
    data = response.json()
    assert "collaterals" in data.keys()

    def test_get_cross_chain_collateral():
        client = TestClient(app)
        COLLATERAL_DATA = [
            {
                "chain_id": "test_chain1",
                "collateral_type": "TokenX",
                "amount": 100,
                "timestamp": datetime.now(),
            },
            {
                "chain_id": "test_chain2",
                "collateral_type": "AssetY",
                "amount": 150,
                "timestamp": datetime.now(),
            },
        ]
        with open("collateral_data.json", "w") as f:
            json.dump(COLLATERAL_DATA, f)
            response = client.get("/chains/test_chain1/collaterals/TokenX")
            assert response.status_code == 200
            data = response.json()
            assert "collateral" in data.keys()

            def test_get_cross_chain_collateral_not_found():
                client = TestClient(app)
                with pytest.raises(HTTPException):
                    response = client.get("/chains/test_chain1/collaterals/TokenX")
                    assert response.status_code == 404


app = FastAPI()


class LiquidityRoutingRequest(BaseModel):
    asset: str
    amount: float

    @app.post("/liquidity-routing")
    async def route_liquidity(request_data: LiquidityRoutingRequest):
        liquidity_data = {
            "BTC": {"price": 35000, "volume": 1000000},
            "ETH": {"price": 2200, "volume": 5000000},
        }
        requested_asset = liquidity_data[request_data.asset]
        optimized_route = "ETH" if request_data.amount > 5000000 else "BTC"
        return {
            "requested_asset": requested_asset["asset"],
            "requested_amount": request_data.amount,
            "optimized_route": optimized_route,
            "optimization_time": time.time(),
        }

    def test_route_liquidity():
        client = TestClient(app)
        data = LiquidityRoutingRequest(asset="ETH", amount=2500000)
        response = client.post("/liquidity-routing", json=data)
        assert response.status_code == 200
        result = response.json()
        expected_asset = "ETH"
        expected_route = "ETH"
        assert result["requested_asset"] == expected_asset
        assert result["optimized_route"] == expected_route

        def test_route_liquidity_large_amount():
            client = TestClient(app)
            data = LiquidityRoutingRequest(asset="BTC", amount=1500000000)
            response = client.post("/liquidity-routing", json=data)
            assert response.status_code == 200
            result = response.json()
            expected_asset = "BTC"
            expected_route = "BTC"
            assert result["requested_asset"] == expected_asset
            assert result["optimized_route"] == expected_route

            def test_route_liquidity_invalid_data():
                client = TestClient(app)
                data = LiquidityRoutingRequest(asset="invalid", amount=100)
                response = client.post("/liquidity-routing", json=data)
                assert response.status_code == 400
                result = response.json()
                error_message = "Invalid input"
                assert "detail" in result
                assert result["detail"]["type"] == "value_error"
                assert error_message in result["detail"]["msg"]


@pytest.fixture()
def client():
    with TestClient(app) as tc:
        yield tc

        def test_create_validation(client):
            response = client.post(
                "/create-validation",
                json={
                    "origin_chain": "test_origin",
                    "destination_chain": "test_destination",
                    "amount": 100,
                },
            )
            assert response.status_code == 200
            assert isinstance(response.json(), dict)
            assert "validation_id" in response.json()

            def test_get_last_validation(client):
                response = client.get("/get-last-validation")
                assert response.status_code == 200
                assert isinstance(response.json(), dict)
                assert "last_validation_id" in response.json()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

        def test_get_market_depth_endpoint(client):
            response = client.get("/market-depth/BTCEUR")
            assert response.status_code == 200
            data = response.json()
            assert "best_bid" in data
            assert "best_ask" in data

            def test_get_market_depth_endpoint_not_found(client):
                response = client.get("/market-depth/INVALID_SYMBOL")
                assert response.status_code == 404


def test_get_bids():
    client = TestClient(app)
    response = client.get("/bids")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

    def test_get_asks():
        client = TestClient(app)
        response = client.get("/asks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc

        def test_historical_data(client):
            response = client.get(
                "/historical_data",
                params={
                    "start_date": "2022-01-01",
                    "end_date": "2022-01-02",
                    "interval": "1min",
                },
            )
            assert response.status_code == 200
            expected_output = list(HistoricalData.parse_raw(response.text))
            assert len(expected_output) > 0, "No historical data returned."


@pytest.mark.anyio
async def test_get_active_orders():
    response = await app.test_client().get("/orders/user/123")
    active_orders = Order.parse_raw(response.content.decode())
    assert len(active_orders) == 2
    assert active_orders[0].id == 1
    assert active_orders[1].id == 2

    def test_get_active_orders_invalid_user_id():
        with pytest.raises(HTTPException):
            response = app.test_client().get("/orders/user/999")
            data = response.json()
            assert data["status"] == "422 Unprocessable Entity"


@pytest.fixture(autouse=True)
def setup(request):
    client = TestClient(app)
    limiter = RateLimiter(max_requests_per_second=10.0, window_seconds=60)

    def teardown():
        if "limiter" in locals():
            del limeter
            request.addfinalizer(teardown)
            request.addfinalizer(teardown)
            yield client
            teardown()

            def test_endpoint(client):
                response = client.get("/endpoint")
                assert response.status_code == 200

                def test_too_many_requests(client):
                    for i in range(11):
                        response = client.get(f"/endpoint?limit={i + 1}")
                        if response.status_code == 429:
                            break
                    else:
                        response = client.get("/endpoint")
                        assert response.status_code == 200
                        assert any(response.status_code == 429)


@pytest.fixture()
def client():
    with TestClient(app) as tc:
        yield tc

        def test_audit_log_endpoint(client):
            response = client.get("/audit-log")
            assert response.status_code == 200
            content = response.json()
            assert "id" in content
            assert "timestamp" in content
            assert "action" in content
            assert "resource" in content
            assert "user_id" in content

            def test_audit_log_empty(client):
                with pytest.raises(HTTPException) as ex:
                    client.get("/audit-log")
                    assert str(ex.value) == "Audit log data is empty."


@pytest.mark.asyncio
async def test_create_recurring_order():
    client = TestClient(app)
    schedule = {"interval": "minute", "count": 10}
    order_data = RecurringOrder(
        symbol="AAPL", quantity=1, price=100.0, schedule=schedule
    )
    response = client.post("/recurring-order", json=order_data.dict())
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "recurring order created" in data["result"]

    def test_create_recurring_order_invalid():
        client = TestClient(app)
        schedule = {"interval": "minute", "count": 10}
        order_data = RecurringOrder(
            symbol="invalid_symbol", quantity=1, price=100.0, schedule=schedule
        )
        response = client.post("/recurring-order", json=order_data.dict())
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid data received" in data["detail"]


def test_post_order_batch():
    client = TestClient(app)
    order_batch = OrderBatch(
        id=str(uuid.uuid4()),
        trading_pairs=["BTC-USD", "ETH-USD"],
        orders=[],
        timestamp=datetime.now(),
    )
    response = client.post("/order-batch", json=order_batch.dict())
    assert response.status_code == 200
    assert "message" in response.json()
    assert "order_batch_id" in response.json()

    def test_get_order_batches():
        client = TestClient(app)
        order_batch1 = OrderBatch(
            id=str(uuid.uuid4()),
            trading_pairs=["BTC-USD", "ETH-USD"],
            orders=[],
            timestamp=datetime.now(),
        )
        order_batch2 = OrderBatch(
            id=str(uuid.uuid4()),
            trading_pairs=["LTC-USD", "XRP-USD"],
            orders=[],
            timestamp=datetime.now(),
        )
        response = client.post("/order-batch", json=order_batch1.dict())
        assert response.status_code == 200
        response = client.get("/order-batches")
        assert response.status_code == 200
        assert "order_batches" in response.json()
        assert len(response.json()["order_batches"]) == 2


@pytest.fixture
def client():
    with TestClient(app) as _app:
        yield _app

        def test_post_bulk_withdrawal_approvals(client):
            data = [
                WithdrawalApproval(
                    approval_id=1,
                    admin_user_id=1,
                    withdrawal_id=2,
                    status="PENDING",
                    approved_at=datetime.now(),
                )
                for _ in range(5)
            ]
            response = client.post("/bulk-withdrawal-approvals", json=data)
            assert response.status_code == 200
            assert isinstance(response.json(), dict)
            assert "message" in response.json()


def test_margin_health_stream():
    client = TestClient(app)
    response = client.get("/ws/margin-health")
    assert response.status_code == 101
    assert b"Upgrade" in response.content
    assert b"Sec-WebSocket-Accept" in response.content

    @pytest.mark.asyncio
    async def test_websocket_endpoint():
        client = TestClient(app)
        async with client.websocket_connect("/ws/margin-health"):
            await client.send_text("")
            msg = await client.receive_json()
            assert "margin_percentage" in msg
            assert client.is_connected() is False


@pytest.fixture()
def client():
    with TestClient(app) as TC:
        yield TC

        def test_identify_wash_trading(client):
            trades = pd.read_csv("trades_data.csv")
            np.random.shuffle(trades)

            @pytest.fixture()
            def api_client():
                return client

            response = client.get("/identify_wash_trading")
            assert response.status_code == 200
            data = response.json()
            assert len(data["trades_detected"]) > 0


@pytest.fixture()
def client():
    yield TestClient(app)

    def test_get_staking_pools(client):
        response = client.get("/staking-pools")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) > 0

        def test_update_reward_distribution_invalid_pool_id(client):
            response = client.put("/update-reward-distribution/pool123")
            assert response.status_code == 400
            assert "Pool ID not found" in str(response.content)

            def test_update_reward_distribution_valid_pool_id(client):
                response = client.put("/update-reward-distribution/pool1")
                assert response.status_code == 200
                assert "Reward distribution updated successfully." in str(
                    response.content
                )


@pytest.fixture
def client():
    with TestClient(app) as _app:
        yield _app

        def test_optimize_fee_tier(client):
            response = client.get("/optimize-feetier")
            data = response.json()
            assert response.status_code == 200
            assert "optimized_fee_tiers" in data.keys()


def test_mirror_position():
    client = TestClient(app)
    data = {"key": "value"}
    expected_response_content = {
        "result": "Position mirrored successfully",
        "data": data,
    }
    response = client.post("/mirror_position", json=data)
    assert response.status_code == 200
    assert response.json() == expected_response_content


@pytest.fixture
def client():
    return TestClient(app)


def test_create_transaction(client):
    response = client.post(
        "/transactions", json={"sender": "Alice", "receiver": "Bob", "amount": 100}
    )
    assert response.status_code == 200
    new_transaction: Transaction = response.json()
    assert new_transaction.sender == "Alice"
    assert new_transaction.receiver == "Bob"
    assert new_transaction.amount == 100

    def test_get_transactions(client):
        response = client.post(
            "/transactions", json={"sender": "Alice", "receiver": "Bob", "amount": 100}
        )
        response = client.get("/transactions")
        transactions: Transaction = response.json()
        assert len(transactions) > 0


def test_validate_oracle_data_success():
    data = OracleData(chain="ethereum", timestamp=datetime(2022, 1, 1), value=100)
    assert validate_oracle_data(data) == {"result": "validation succeeded"}

    def test_validate_oracle_data_unsupported_chain():
        data = OracleData(chain="unsupported_chain")
        with pytest.raises(HTTPException):
            _ = validate_oracle_data(data)
            assert True

            def test_validate_oracle_data_invalid_timestamp():
                data = OracleData(
                    chain="ethereum", timestamp=datetime(2100, 1, 1), value=100
                )
                with pytest.raises(HTTPException):
                    _ = validate_oracle_data(data)
                    assert True


@pytest.main
def test_debt_position_investor():
    investor = DebtPosition.Investor(
        id=1, name="John Doe", email="john.doe@example.com"
    )
    dp = DebtPosition(id=1, amount=1000.0, investor_id=investor.id, interest_rate=0.05)
    assert dp.investor_id == investor.id


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_generate_wallet_address_with_all_parameters(client):
    response = await client.post(
        "/wallet-address", json={"currency": "BTC", "user_id": 123456}
    )
    assert response.status_code == 200
    data = await response.json()
    assert "user_id" in data
    assert "currency" in data
    assert "wallet_address" in data
    assert "creation_date" in data

    @pytest.mark.asyncio
    async def test_generate_wallet_address_with_missing_parameters(client):
        response = await client.post("/wallet-address", json={})
        assert response.status_code == 422

        @pytest.mark.asyncio
        async def test_generate_wallet_address_with_optional_creation_date(client):
            response = await client.post(
                "/wallet-address", json={"currency": "BTC", "user_id": 123456}
            )
            assert response.status_code == 200
            data = await response.json()
            assert "user_id" in data
            assert "currency" in data
            assert "wallet_address" in data
            assert isinstance(data["creation_date"], str)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_generate_wallet_address_with_all_parameters(client):
    response = await client.post(
        "/wallet-address", json={"currency": "BTC", "user_id": 123456}
    )
    assert response.status_code == 200
    data = await response.json()
    assert "user_id" in data
    assert "currency" in data
    assert "wallet_address" in data
    assert "creation_date" in data

    @pytest.mark.asyncio
    async def test_generate_wallet_address_with_missing_parameters(client):
        response = await client.post("/wallet-address", json={})
        assert response.status_code == 422

        @pytest.mark.asyncio
        async def test_generate_wallet_address_with_optional_creation_date(client):
            response = await client.post(
                "/wallet-address", json={"currency": "BTC", "user_id": 123456}
            )
            assert response.status_code == 200
            data = await response.json()
            assert "user_id" in data
            assert "currency" in data
            assert "wallet_address" in data
            assert isinstance(data["creation_date"], str)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


class TestWalletGeneration(unittest.TestCase):

    @patch("app.generate_wallet")
    def test_generate_wallet_with_bothParameters(self, mock_func):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func("BTC", "test").return_value = (mock_response, b"")
        client = patch("http.client.Client")
        client_instance = client.start()
        result = client_instance.get("/generate-wallet", "BTC", "test")
        self.assertEqual(200, result.status_code)

        def test_generate_wallet_without_parameters(self):
            response = unittest.mock.Mock()
            response.status_code = 400
            client = patch("http.client.Client")
            client_instance = client.start()
            result = client_instance.get("/generate-wallet")
            self.assertEqual(400, result.status_code)

            def test_generate_wallet_only_currency(self):
                response = unittest.mock.Mock()
                response.status_code = 400
                client = patch("http.client.Client")
                client_instance = client.start()
                result = client_instance.get("/generate-wallet", "BTC")
                self.assertEqual(400, result.status_code)

                def test_generate_wallet_only_username(self):
                    response = unittest.mock.Mock()
                    response.status_code = 400
                    client = patch("http.client.Client")
                    client_instance = client.start()
                    result = client_instance.get("/generate-wallet", None, "test")
                    self.assertEqual(400, result.status_code)


@pytest.mark.asyncio
async def test_authenticate_token_successfully():
    client = TestClient(app)
    response = await client.get("/auth")
    assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_authenticate_token_unauthorized():
        client = TestClient(app)
        response = await client.get(
            "/auth", headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401


@pytest.fixture
def client():
    return TestClient(app)


def test_check_wallet_with_wallet(client):
    """Test endpoint with a provided wallet address"""
    test_wallet = "ABC123"
    response = client.get(f"/check-wallet?wallet={test_wallet}")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "ABC123" in data and data["ABC123"] == 5000

    def test_check_wallet_without_wallet(client):
        """Test endpoint without providing a wallet address"""
        response = client.get("/check-wallet")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict) or data is None
        if not data:
            assert "Error" in data.get("error", "")

            def test_check_wallet_empty_client(client):
                """Test that endpoint doesn't crash with empty request"""
                try:
                    client.get("/check-wallet")
                except Exception as e:
                    pytest.fail(f"Unexpected error occurred: {str(e)}")


client = TestClient(api)


@pytest.mark.asyncio
async def test_successful_order_placement():
    """Test placing a valid order with all required parameters."""
    data = {
        "trade_pair": "BTC/USDT",
        "amount": 1.0,
        "stop_price": None,
        "venue": "exchange",
    }
    response = await client.post("/test", json=data)
    assert response.status_code == 200
    response_data = await response.json()
    assert "order_id" in response_data

    @pytest.mark.asyncio
    async def test_order_without_trade_pair():
        """Test error handling when 'trade_pair' is missing."""
        data = {"amount": None, "stop_price": None, "venue": None}
        with pytest.raises(Exception) as excinfo:
            await client.post("/test", json=data)
            assert str(excinfo.value) == "missing 'trade_pair' required"

            @pytest.mark.asyncio
            async def test_order_without_venue():
                """Test error handling when 'venue' is missing."""
                data = {"trade_pair": "BTC/USDT", "amount": 1.0, "stop_price": None}
                with pytest.raises(Exception) as excinfo:
                    await client.post("/test", json=data)
                    assert str(excinfo.value) == "missing 'venue' required"

                    @pytest.mark.asyncio
                    async def test_order_with_invalid_amount_type():
                        """Test error handling when 'amount' is not a valid number."""
                        data = {
                            "trade_pair": "BTC/USDT",
                            "amount": "1.0",
                            "stop_price": None,
                            "venue": "exchange",
                        }
                        with pytest.raises(Exception) as excinfo:
                            await client.post("/test", json=data)
                            assert str(excinfo.value).startswith(
                                "could not convert string to float"
                            )

                            @pytest.mark.asyncio
                            async def test_order_with_invalid_venue():
                                """Test error handling when 'venue' is an invalid or missing value."""
                                data = {
                                    "trade_pair": "BTC/USDT",
                                    "amount": 1.0,
                                    "stop_price": None,
                                    "venue": None,
                                }
                                with pytest.raises(Exception) as excinfo:
                                    await client.post("/test", json=data)
                                    assert (
                                        str(excinfo.value) == "missing 'venue' required"
                                    )

                                    @pytest.mark.asyncio
                                    async def test_order_with_valid_data():
                                        """Test a completely valid order with all required parameters."""
                                        data = {
                                            "trade_pair": "ETH/USDT",
                                            "amount": 0.5,
                                            "stop_price": 30000,
                                            "venue": "exchange",
                                        }
                                        response = await client.post("/test", json=data)
                                        assert response.status_code == 200
                                        response_data = await response.json()
                                        assert isinstance(response_data, dict)
                                        assert "order_id" in response_data


@pytest.fixture
def order_book() -> list:
    return [
        {"price": 100.0, "quantity": 5},
        {"price": 98.0, "quantity": 3},
        {"price": 102.0, "quantity": 4},
    ]


@pytest.fixture
def order_book_updated(order_book: list) -> list:
    return [
        {"price": 100.0, "quantity": 5},
        {"price": 98.0, "quantity": 3},
        {"price": 102.0, "quantity": 4},
    ]


@pytest.mark.parametrize(
    "price, quantity", [(101.0, 5), (98.0, 3), (102.0, 4), (110.0, 5)]
)
def test_place_limit_sell_order_success(
    client: TestClient,
    order_book: list,
    order_book_updated: list,
    price: float,
    quantity: int,
):
    """Test placing a limit sell order with various prices and quantities"""
    new_order = {
        "price": price,
        "quantity": quantity,
        "current_time": datetime.now().time(),
    }
    if any((order["price"] == price for order in order_book)):
        order = [order for order in order_book if order["price"] == price][0]
        order["quantity"] += quantity
        response = client.post("/limit-sell-order", json=new_order)
    else:
        order_book_copy = order_book.copy()
        order_book_copy.insert(0, new_order)
        response = client.post("/limit-sell-order", json=new_order)
        assert response.status_code == 200
        if any((order["price"] == price for order in order_book)):
            assert "Order placed successfully" in str(response.json())
            assert all(
                (
                    order["price"] != new_order["price"] or order["quantity"] > quantity
                    for order in order_book_copy
                )
            )
        else:
            assert any(
                (order["price"] == price for order in response.json()["order_book"])
            )
            order_found = [
                order
                for order in response.json()["order_book"]
                if order["price"] == price
            ][0]
            assert (
                "quantity" in order_found and int(order_found["quantity"]) == quantity
            )

            def test_place_limit_sell_order_same_price(
                client: TestClient, order_book: list
            ):
                """Test updating an existing order at the same price"""
                new_order = {
                    "price": 100.0,
                    "quantity": 5,
                    "current_time": datetime.now().time(),
                }
                order_book_copy = [order.copy() for order in order_book]
                response = client.post("/limit-sell-order", json=new_order)
                assert response.status_code == 200
                existing_order = None
                for order in order_book_copy:
                    if order["price"] == new_order["price"]:
                        existing_order = order.copy()
                        break
                else:
                    assert response.json()["order_book"][0]["price"] != 100.0
                    if existing_order:
                        assert existing_order["quantity"] == new_order["quantity"]
                        assert "Order placed successfully" in str(response.json())

                        def test_place_limit_sell_order_no_existing_price(
                            client: TestClient, order_book: list
                        ):
                            """Test placing an order with a price that doesn't exist"""
                            new_order = {
                                "price": 95.0,
                                "quantity": 2,
                                "current_time": datetime.now().time(),
                            }
                            response = client.post("/limit-sell-order", json=new_order)
                            assert response.status_code == 200
                            order_book_copy = [order.copy() for order in order_book]
                            existing_orders_at_95 = sum(
                                (
                                    order["quantity"]
                                    for order in order_book_copy
                                    if order["price"] == 95.0
                                )
                            )
                            assert new_order not in order_book_copy
                            assert "Order placed successfully" in str(response.json())
                            assert len(order_book_copy) + 1 == len(
                                response.json()["order_book"]
                            )

                            def test_place_limit_sell_order_multiple_orders(
                                client: TestClient, order_book: list
                            ):
                                """Test handling multiple orders at the same price"""
                                new_order = {
                                    "price": 98.0,
                                    "quantity": 5,
                                    "current_time": datetime.now().time(),
                                }
                                response = client.post(
                                    "/limit-sell-order", json=new_order
                                )
                                assert response.status_code == 200
                                order_book_copy = [order.copy() for order in order_book]
                                existing_orders_at_98 = sum(
                                    (
                                        order["quantity"]
                                        for order in order_book_copy
                                        if order["price"] == 98.0
                                    )
                                )
                                new_quantity = (
                                    new_order["quantity"] + existing_orders_at_98
                                )
                                assert order_book_copy[1]["quantity"] == new_quantity
                                assert "Order placed successfully" in str(
                                    response.json()
                                )


class Order(BaseModel):
    id: str
    symbol: str
    quantity: float
    price: float
    side: str

    @pytest.fixture
    def client():
        return TestClient(app)

    def test_get_order_book(client):
        buy_orders = [
            Order(
                id=f"new_buy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol="BTC",
                quantity=1.0,
                price=45000,
                side="buy",
            )
        ]
        sell_orders = [
            Order(
                id=f"new_sell_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol="ETH",
                quantity=2.0,
                price=30000,
                side="sell",
            )
        ]
        response = client.get("/order_book")
        assert response.status_code == 200
        response_data = json.loads(response.content)
        assert "buy" in response_data and "sell" in response_data
        assert len(response_data["buy"]) > 0
        assert len(response_data["sell"]) > 0

        def test_match_limit_order(client):
            buy_orders = []
            sell_orders = []
            response = client.post(
                "/match_order",
                json={"symbol": "BTC", "quantity": 1.0, "price": 45000, "side": "buy"},
            )
            assert response.status_code == 200
            response_data = json.loads(response.content)
            assert "filled" in response_data and "matched" in response_data
            assert len(response_data["filled"]) == 0

            def test_filled_orders(client):
                buy_orders = [
                    Order(
                        id=f"new_buy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        symbol="BTC",
                        quantity=1.0,
                        price=45000,
                        side="buy",
                    )
                ]
                sell_orders = [
                    Order(
                        id=f"new_sell_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        symbol="ETH",
                        quantity=2.0,
                        price=30000,
                        side="sell",
                    )
                ]
                response = client.post("/filled_orders")
                assert response.status_code == 200
                response_data = json.loads(response.content)
                assert "filled" in response_data
                assert len(response_data["filled"]) > 0


@pytest.fixture
def client():
    return TestClient(app)


def test_get_order_book_alpha(client):
    """Test that GET /order_book/alpha returns a 200 status code."""
    response = client.get("/order_book/alpha")
    assert response.status_code == 200

    def test_invalid_request_method(client):
        """Test that invalid HTTP methods return the correct status code."""
        response = client.post("/order_book/alpha")
        assert response.status_code == 405

        def test_missing_query_parameter(client):
            """Test that missing query parameter returns 400 status code."""
            response = client.get("/order_book", query_params={"symbol": "invalid"})
            assert response.status_code == 400
            response.json()

            def test_invalid_symbol(client):
                """Test that invalid symbol returns the correct status code."""
                response = client.get("/order_book/abc")
                assert response.status_code == 200


@pytest.fixture
def websocket_client():
    with patch("fastapi.testclient.TestClient") as client:
        yield client

        async def test_websocket_message(ws: WsClient):
            """Test receiving a message via WebSocket handler."""
            await ws.send("test-message")
            with pytest.raises(Exception) as excinfo:
                await ws.get("/ws/trade-pairs")
                assert "Message received" in str(excinfo.value)
                assert "test-message" in str(excinfo.value)

                async def test_websocket_endpoint(websocket_client: TestClient):
                    """Test the WebSocket endpoint functionality."""
                    try:
                        await websocket_client.get("/ws/trade-pairs")
                        response = await websocket_client.get("/ws/trade-pairs").json()
                        assert isinstance(response, dict)
                        assert "message" in response
                    except Exception as e:
                        print(f"Error testing WebSocket endpoint: {e}")


app = FastAPI()


@pytest.fixture
def test_client():
    return TestClient(app)


def test_get_historical_data正常响应(test_client):
    """
    测试/historical_data endpoint返回正常的响应数据
    """
    symbol = "AAPL"
    start_date = datetime(2023, 1, 1).isoformat()
    end_date = datetime(2023, 12, 31).isoformat()
    interval = "1d"
    try:
        response = test_client.post(
            "/historical_data",
            symbol=symbol,
            start=start_date,
            end=end_date,
            interval=interval,
        )
        assert response.status_code == 200
        assert "data" in response.json()
    except Exception as e:
        pytest.fail(f"Request failed: {str(e)}")

        def test_get_historical_data_empty_time_period(test_client):
            """
            测试提供无效的时间范围（start > end）导致的错误响应
            """
            symbol = "AAPL"
            start_date = "2023-12-31"
            end_date = "2023-01-01"
            try:
                response = test_client.post(
                    "/historical_data", symbol=symbol, start=start_date, end=end_date
                )
                assert response.status_code == 500
            except Exception as e:
                pytest.fail(f"Request failed: {str(e)}")

                def test_get_historical_data_invalid_symbol(test_client):
                    """
                    测试提供无效的股票符号导致的错误响应
                    """
                    symbol = "XYZX"
                    start_date = "2023-01-01"
                    end_date = "2023-12-31"
                    try:
                        response = test_client.post(
                            "/historical_data",
                            symbol=symbol,
                            start=start_date,
                            end=end_date,
                        )
                        assert response.status_code == 500
                    except Exception as e:
                        pytest.fail(f"Request failed: {str(e)}")

                        def test_get_historical_data_invalid_interval(test_client):
                            """
                            测试提供无效的时间区间导致的错误响应
                            """
                            symbol = "AAPL"
                            start_date = "2023-01-01"
                            end_date = "2023-12-31"
                            try:
                                response = test_client.post(
                                    "/historical_data",
                                    symbol=symbol,
                                    start=start_date,
                                    end=end_date,
                                    interval="1y",
                                )
                                assert response.status_code == 500
                            except Exception as e:
                                pytest.fail(f"Request failed: {str(e)}")


@pytest.fixture
def client():
    with Session() as db:
        yield db

        @pytest.fixture
        def db_client(db):
            with FastAPI.testappTestClient(app, db) as client:
                yield client

                async def test_successful_withdraw():
                    async with db_client:
                        async with suppress(Exception):
                            response = await app.get(
                                "/api/v1/withdraw/test-address",
                                {"amount": 100, "user": {"balance": 200}},
                            )
                            assert response.status_code == 200

                            async def test_inufficient_balance():
                                async with db_client:
                                    async with suppress(Exception):
                                        response = await app.get(
                                            "/api/v1/withdraw/test-address",
                                            {"amount": 300, "user": {"balance": 200}},
                                        )
                                        assert response.status_code == 403

                                        async def test_unauthorized_withdraw():
                                            async with db_client:
                                                user = {
                                                    "balance": 500,
                                                    "isAdmin": False,
                                                }
                                                async with suppress(HTTPException):
                                                    response = await app.get(
                                                        "/api/v1/withdraw/test-address",
                                                        {"amount": 200, "user": user},
                                                    )
                                                    assert response.status_code == 403


@pytest.fixture
def client():
    return TestClient(app)


def test_get_active_orders_success(client):
    user = User(email="test@example.com", username="testuser", password="testpass")
    current_date = datetime.now()
    order1 = Order(
        user_id=1,
        customer_id=1,
        product_id=1,
        quantity=1,
        price=10.0,
        status="active",
        created_at=current_date,
        updated_at=current_date,
    )
    order2 = Order(
        user_id=1,
        customer_id=2,
        product_id=2,
        quantity=2,
        price=5.0,
        status="active",
        created_at=current_date,
        updated_at=current_date,
    )
    User.query.delete()
    User(user=user).add()
    Order.query.delete()
    order1.add()
    order2.add()
    client.get("/orders/1")

    def test_get_active_orders_success():
        pass

        def test_get_empty_orders(client):
            response = client.get("/orders/1")
            assert response.status_code == 404
            assert "Not Found" in response.content.decode()

            def test_get_orders_with_correct_model(client):
                response = client.get("/orders/1")
                assert isinstance(response.json()["orders"], list)
                for order in response.json()["orders"]:
                    assert isinstance(order, dict)

                    def test_get_nonexistent_user(client):
                        response = client.get("/orders/9999")
                        assert response.status_code == 404


@pytest.fixture
def app():
    app = FastAPI()
    redis_key = mock.Mock()
    redisip = mock.Mock()

    @app.get("/api/key/{key}/limit")
    async def api_key_limiting(key: str):
        pass

        @app.get("/api/ip/{ip}/limit")
        async def ip_address_limiting(ip: str):
            pass

            @app.get("/endpoint")
            def endpoint(**kwargs):
                return {"result": "value"}

            return app

        @pytest.fixture
        def client(app):
            return TestClient(app)

        @pytest.mark.asyncio
        async def test_endpoint(client):
            response = await client.get("/endpoint")
            assert response.status_code == 200

            @pytest.mark.asyncio
            async def test_key_limit(client):
                response = await client.get(
                    "/api/key/test/limit", headers={"x-api-key": "test"}
                )
                assert response.status_code == 500
                assert "Exceeded API key limit" in str(response.json())

                @pytest.mark.asyncio
                async def test_ip_limit(client):
                    response = await client.get(
                        "/api/ip/1.2.3.4/limit", headers={"x-forwarded-ip": "1.2.3.4"}
                    )
                    assert response.status_code == 500
                    assert "Exceeded IP address limit" in str(response.json())

                    @pytest.mark.asyncio
                    async def test_rate_limiting(client):

                        async def mock_redis_key_get(key: str) -> bytes:
                            return b"limit_exceeded"

                        async def mock_redis_incr(key: str) -> int:
                            return 2

                        with patch.getmockarry(
                            redis_key.get, "mock_redis_key_get"
                        ) as m:
                            m.return_value = b"limit_exceeded"
                            response = await client.get(
                                "/endpoint", headers={"x-api-key": "test"}
                            )
                            assert response.status_code == 500
                            assert "key_limit_exceeded" in response.json()
                            response = await client.get(
                                "/endpoint", headers={"x-api-key": "test"}
                            )
                            assert response.status_code == 429
                            mock_redis_incr.assert_called_once_with("test")


@pytest.fixture
def client():
    return TestClient(app)


def test_cart_remove_success(client):
    response = client.post("/cart/remove", {"remove": True, "id": 123})
    assert response.status_code == 200
    assert "message" in response.json()
    assert isinstance(response.json()["message"], str)

    def test_cart_remove_missing_remove(client):
        response = client.post("/cart/remove", {"id": 123})
        assert response.status_code == 400
        assert "detail" in response.json()
        assert response.json()["detail"] == "Invalid parameters"

        def test_orders_cancel_success(client):
            response = client.post("/orders/123", {"remove": True})
            assert response.status_code == 200
            assert "message" in response.json()
            assert isinstance(response.json()["message"], str)

            def test_orders_cancel_missing_order_id(client):
                response = client.post("/orders")
                assert response.status_code == 400
                assert "detail" in response.json()
                assert response.json()["detail"] == "No valid order_id provided"

                def test_orders_cancel_missing_remove(client):
                    response = client.post("/orders/123")
                    assert response.status_code == 400
                    assert "detail" in response.json()
                    assert (
                        response.json()["detail"] == "No removal confirmation received"
                    )


def create_transaction(
    sender: str = "Unknown", receiver: str = "Unknown", amount: float = 1.0
) -> dict:
    return {
        "id": str(uuid4()),
        "date": datetime.now().isoformat(),
        "sender": sender,
        "receiver": receiver,
        "amount": amount,
        "status": TransactionStatus.PENDING,
    }


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.asyncio
async def test_get_balance_success(client: TestClient):
    """Test that the /balance endpoint returns the current balance."""
    response = await client.get("/balance")
    assert response.status_code == 200
    data = await response.json()
    assert "balance" in app.state
    assert data["balance"] == app.state.balance

    @pytest.mark.asyncio
    async def test_get_transactions(client: TestClient):
        """Test that the /transactions endpoint returns a list of transactions."""
        response = await client.get("/transactions")
        assert response.status_code == 200
        data = await response.json()
        assert isinstance(data, List)
        assert len(data) > 0
        transaction = data[0]
        assert "id" in transaction
        assert "date" in transaction
        assert "sender" in transaction
        assert "receiver" in transaction
        assert "amount" in transaction
        assert "status" in transaction

        @pytest.mark.asyncio
        async def test_deposit_success(client: TestClient):
            """Test that a successful deposit increases the balance."""
            response = await client.post("/deposit", json=10.0)
            assert response.status_code == 200
            assert "message" in response.json()
            assert "balance" in response.json()
            assert app.state.balance > 0

            @pytest.mark.asyncio
            async def test_deposit_invalid_amount(client: TestClient):
                """Test that depositing a non-positive amount raises an error."""
                with pytest.raises(HTTPException) as exc_info:
                    await client.post("/deposit", json=-1.0)
                    assert exc_info.value.status_code == 400
                    assert "detail" in exc_info.value.json()
                    assert "Amount must be positive." in exc_info.value.json()
                    if __name__ == "__main__":
                        pytest.main()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_generate_key_with_all_parameters(client):
    response = await client.get(
        "/api/generate-key",
        username="testuser",
        email="test@example.com",
        password="testpass",
        include_read=True,
        include_write=False,
        include_admin=True,
    )
    assert response.status_code == 200
    data = response.json()
    assert "api_key" in data
    assert "username" in data
    assert "email" in data
    assert "role" in data
    assert "created_at" in data
    assert "expires_at" in data

    @pytest.mark.asyncio
    async def test_include_read_only(client):
        response = await client.get(
            "/api/generate-key",
            username="testuser",
            email="test@example.com",
            password="testpass",
            include_read=True,
            include_write=False,
            include_admin=False,
        )
        assert response.status_code == 200
        data = response.json()
        role = "Read"
        assert role in data["role"]
        assert not any((s in data["role"] for s in ["Write", "Admin"]))

        @pytest.mark.asyncio
        async def test_include_write_only(client):
            response = await client.get(
                "/api/generate-key",
                username="testuser",
                email="test@example.com",
                password="testpass",
                include_read=False,
                include_write=True,
                include_admin=False,
            )
            assert response.status_code == 200
            data = response.json()
            role = "Write"
            assert role in data["role"]
            assert not any((s in data["role"] for s in ["Read", "Admin"]))

            @pytest.mark.asyncio
            async def test_include_admin_only(client):
                response = await client.get(
                    "/api/generate-key",
                    username="testuser",
                    email="test@example.com",
                    password="testpass",
                    include_read=False,
                    include_write=False,
                    include_admin=True,
                )
                assert response.status_code == 200
                data = response.json()
                role = "Admin"
                assert role in data["role"]
                assert not any((s in data["role"] for s in ["Read", "Write"]))

                @pytest.mark.asyncio
                async def test_include_read_and_write(client):
                    response = await client.get(
                        "/api/generate-key",
                        username="testuser",
                        email="test@example.com",
                        password="testpass",
                        include_read=True,
                        include_write=True,
                        include_admin=False,
                    )
                    assert response.status_code == 200
                    data = response.json()
                    role = "Read Write"
                    assert role in data["role"]
                    assert not "Admin" in data["role"]

                    @pytest.mark.asyncio
                    async def test_include_read_and_admin(client):
                        response = await client.get(
                            "/api/generate-key",
                            username="testuser",
                            email="test@example.com",
                            password="testpass",
                            include_read=True,
                            include_write=False,
                            include_admin=True,
                        )
                        assert response.status_code == 200
                        data = response.json()
                        role = "Read Admin"
                        assert role in data["role"]
                        assert not "Write" in data["role"]

                        @pytest.mark.asyncio
                        async def test_include_write_and_admin(client):
                            response = await client.get(
                                "/api/generate-key",
                                username="testuser",
                                email="test@example.com",
                                password="testpass",
                                include_read=False,
                                include_write=True,
                                include_admin=True,
                            )
                            assert response.status_code == 200
                            data = response.json()
                            role = "Write Admin"
                            assert role in data["role"]
                            assert not "Read" in data["role"]

                            @pytest.mark.asyncio
                            async def test_empty_parameters(client):
                                with pytest.raises(
                                    ValueError,
                                    match="Username, Email, and Password are required",
                                ):
                                    client.get("/api/generate-key")

                                    @pytest.mark.asyncio
                                    async def test_incomplete_parameters(client):
                                        response = await client.get(
                                            "/api/generate-key",
                                            username="testuser",
                                            email=None,
                                            password=None,
                                        )
                                        assert response.status_code == 422

                                        @pytest.mark.asyncio
                                        async def test_empty_username(client):
                                            response = await client.get(
                                                "/api/generate-key",
                                                username="",
                                                email="test@example.com",
                                                password="testpass",
                                            )
                                            data = response.json()
                                            role = "Read"
                                            assert not any(
                                                (
                                                    s in data["role"]
                                                    for s in ["Write", "Admin"]
                                                )
                                            )

                                            @pytest.mark.asyncio
                                            async def test_expires_at(client):
                                                response = await client.get(
                                                    "/api/generate-key",
                                                    username="testuser",
                                                    email="test@example.com",
                                                    password="testpass",
                                                    include_read=True,
                                                    include_write=False,
                                                    include_admin=False,
                                                )
                                                data = response.json()
                                                assert "expires_at" in data
                                                current_time = time.time()
                                                expected_expires = current_time + 3600
                                                assert (
                                                    data["expires_at"]
                                                    > expected_expires - 3600
                                                    and data["expires_at"]
                                                    < expected_expires + 3600
                                                )


@pytest.fixture
def client():
    return TestClient(app)


def test_get_trading_fee_rate_with_volume_over_10000(client):
    user_data = {"username": "test_user", "balance": 10000, "30_day_volume": 15000}
    expected_fee = 0.2
    response = client.get("/trading-fee-rate", user_data)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Trading fee rate calculated successfully"
    assert data["fee_rate"] == expected_fee

    def test_get_trading_fee_rate_with_volume_5000(client):
        user_data = {"username": "test_user", "balance": 10000, "30_day_volume": 5000}
        expected_fee = 0.3
        response = client.get("/trading-fee-rate", user_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trading fee rate calculated successfully"
        assert data["fee_rate"] == expected_fee

        def test_get_trading_fee_rate_with_volume_10000(client):
            user_data = {
                "username": "test_user",
                "balance": 10000,
                "30_day_volume": 10000,
            }
            expected_fee = 0.2
            response = client.get("/trading-fee-rate", user_data)
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Trading fee rate calculated successfully"
            assert data["fee_rate"] == expected_fee

            def test_get_trading_fee_rate_with_volume_under_5000(client):
                user_data = {
                    "username": "test_user",
                    "balance": 10000,
                    "30_day_volume": 4999,
                }
                expected_fee = 0.5
                response = client.get("/trading-fee-rate", user_data)
                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Trading fee rate calculated successfully"
                assert data["fee_rate"] == expected_fee

                def test_get_trading_fee_rate_with_volume_over_10000_with_string_values(
                    client,
                ):
                    user_data = {
                        "username": "test_user",
                        "balance": "10000",
                        "30_day_volume": "15000",
                    }
                    expected_fee = 0.2
                    response = client.get("/trading-fee-rate", user_data)
                    assert response.status_code == 200
                    data = response.json()
                    assert data["message"] == "Trading fee rate calculated successfully"
                    assert data["fee_rate"] == expected_fee

                    def test_get_trading_fee_rate_with_no_volume(client):
                        user_data = {"username": "test_user", "balance": 10000}
                        response = client.get("/trading-fee-rate", user_data)
                        assert response.status_code == 200
                        data = response.json()
                        assert (
                            data["message"]
                            == "Trading fee rate calculated successfully"
                        )
                        assert data["fee_rate"] == 0.5

                        def test_get_trading_fee_rate_with_non_number_volume(client):
                            user_data = {
                                "username": "test_user",
                                "balance": 10000,
                                "30_day_volume": "abc",
                            }
                            response = client.get("/trading-fee-rate", user_data)
                            assert response.status_code == 200
                            data = response.json()
                            assert (
                                data["message"]
                                == "Trading fee rate calculated successfully"
                            )
                            assert data["fee_rate"] == 0.5

                            def test_get_trading_fee_rate_with_missing_data(client):
                                user_data = {
                                    "username": None,
                                    "balance": None,
                                    "30_day_volume": 10000,
                                }
                                response = client.get("/trading-fee-rate", user_data)
                                assert response.status_code == 200
                                data = response.json()
                                assert (
                                    data["message"]
                                    == "Trading fee rate calculated successfully"
                                )
                                assert data["fee_rate"] == 0.3


@pytest.fixture
def client():
    return TestClient(app)


def test_trading_statistics_error(client):
    response = client.get("/trading_statistics")
    assert response.status_code == 200
    assert "error" in response.json()
    assert response.json() != {}

    def test_trading_statistics_single_pair(client):
        bid1 = 100.0
        ask1 = 99.0
        pair1 = {"bid": bid1, "ask": ask1}
        response = client.get(f"/trading_statistics")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        first_pair = data["data"][0]
        assert f"{first_pair['pair']:.4f}" == "100.00.00/99.00.00"
        for key in ["last_price", "highest_price", "lowest_price", "volume"]:
            value = first_pair[key]
            if key != "volume":
                assert isinstance(value, float)
                formatted_value = f"{value:.2f}"
                assert formatted_value == str(value).replace(".", "").ljust(4, "0")[:4]

                def test_trading_statistics_multiple_pairs(client):
                    bid1 = 100.0
                    ask1 = 99.0
                    bid2 = 150.0
                    ask2 = 149.0
                    pair1 = {"bid": bid1, "ask": ask1}
                    pair2 = {"bid": bid2, "ask": ask2}
                    response = client.get(f"/trading_statistics")
                    assert response.status_code == 200
                    data = response.json()
                    assert isinstance(data["data"], list)
                    assert len(data["data"]) == 2
                    first_pair = data["data"][0]
                    second_pair = data["data"][1]
                    assert f"{first_pair['pair']:.4f}" == "100.00.00/99.00.00"
                    assert f"{second_pair['pair']:.4f}" == "150.00.00/149.00.00"
                    last_price = (bid1 + ask1) / 2
                    highest_price = max(bid1, ask1)
                    lowest_price = min(bid1, ask1)
                    volume = bid1 + ask1
                    assert round(first_pair["last_price"], 2) == round(last_price, 2)
                    assert first_pair["highest_price"] == last_price
                    assert first_pair["lowest_price"] == lowest_price
                    assert first_pair["volume"] == volume
                    last_price = (bid2 + ask2) / 2
                    highest_price = max(bid2, ask2)
                    lowest_price = min(bid2, ask2)
                    volume = bid2 + ask2
                    assert round(second_pair["last_price"], 2) == round(last_price, 2)
                    assert second_pair["highest_price"] == last_price
                    assert second_pair["lowest_price"] == lowest_price
                    assert second_pair["volume"] == volume


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///database.db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "maindb")


def get_db_url():
    return f"postgresql://{DB_USER}:{quote(DB_PASSWORD)}@{os.getenv('DB_HOST', 'localhost')}/{DB_NAME}"


@pytest.fixture
def client():
    """Create a FastAPI test client instance."""
    return TestClient(app)


def test_migrate_success(client):
    """Test that the database can be migrated successfully."""
    try:
        response = client.post("/migrate", json={"migrate": 1})
        assert response.status_code == status.HTTP_200_OK
        assert "Migrated successfully" in response.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_ERROR,
            detail=f"Migration failed: {str(e)}",
        )

        def test_schema_endpoint(client):
            """Test that the schema endpoint returns the correct version."""
            try:
                current_head = os.getenv("CURRENT_MIGRATION", "0")
                response = client.get("/schema")
                assert response.status_code == status.HTTP_200_OK
                assert response.json() == {"schema_version": int(current_head)}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_ERROR,
                    detail=f"Schema endpoint failed: {str(e)}",
                )
