# config.py

# Time window considered as "unsafe"
UNSAFE_HOURS = (22, 5)  # 10 PM to 5 AM

# Thresholds
CONFIDENCE_THRESHOLD = 0.7

# Model paths
AGE_MODEL = {
    "proto": "models/age_deploy.prototxt",
    "model": "models/age_net.caffemodel"
}
GENDER_MODEL = {
    "proto": "models/gender_deploy.prototxt",
    "model": "models/gender_net.caffemodel"
}

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
