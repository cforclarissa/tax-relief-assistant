import os
import hashlib
import base64
import helper_functions.llm as llm

def generate_salt(length: int = 16) -> str:
    """Generate a random salt."""
    return base64.b64encode(os.urandom(length)).decode('utf-8')

def hash_password(password: str, salt: str) -> str:
    """Hash a password with a given salt."""
   
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def verify_password(role: str, input_username: str, input_password: str) -> bool:
    """Verify input password against stored hash."""
    retvalue = False
    try:
        msg = llm.getClient(role, input_username)
        if (msg is not None):
            retvalue = hash_password(input_password, msg[0]["SALT"]) == msg[0]["HASH"]
            
    except Exception as e:
        print(f"Security function error: {e}")
    return retvalue