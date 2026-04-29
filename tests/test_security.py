from backend.core.security import get_password_hash, verify_password


def test_password_hash_roundtrip():
    password = "demo-pass-123"
    hashed = get_password_hash(password)

    assert hashed != password
    assert verify_password(password, hashed)
