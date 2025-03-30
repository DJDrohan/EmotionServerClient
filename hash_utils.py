# hash_utils.py


import hashlib
import os


"""
Program Name: hash_utils.py

Author: DJ Drohan

Student Number: C21315413

Date: 01/03/25

Program Description:

Script that uses Hashlib for password salts and hashing

"""

# Password hashing and verification logic
def hash_data(data, salt, iterations=42): #42 hash iterations with salt
    hash_result = (salt + data).encode() #make hashed password
    for _ in range(iterations):
        hash_result = hashlib.sha256(hash_result).digest()  #keep hashing password until iterations is met
    return hash_result.hex()

# Generate salt and hash from the startup password
def generate_password_hash(password):
    salt = os.urandom(16).hex() #random salt generation
    hashed_password = hash_data(password, salt) #call to hash password function
    return salt, hashed_password #returns salt and hashed password to server
