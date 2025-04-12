from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import current_user
from transformers import pipeline
import requests
import numpy as np
import nltk
import spacy
import spacy.cli
import re
from datetime import datetime
from collections import Counter
import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Database configuration
SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'sentiments.db')}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Secret key for Flask sessions and security
SECRET_KEY = os.getenv("SECRET_KEY", "your_default_dev_secret_key")

# Optional: Debug mode (not required unless you're toggling it programmatically)
DEBUG = False
