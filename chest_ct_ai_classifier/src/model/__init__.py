# Make utils a package for reliable imports in tests
from .model_generator import generate_model
from .inference import MedicalModelInference
from .models import resnet
from .model_evaluate import evaluate_model
