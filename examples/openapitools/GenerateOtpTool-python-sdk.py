import json
import sys
import requests

# DONT CHANGE INPUT PART START
try:
    input_json = input_json
except:
    input_json = json.loads(sys.argv[1])
env = input_json.pop('openv', {})
# DONT CHANGE INPUT PART END

try:
    phone_number = input_json.get('phoneNumber', 0)
    # Output the result
    print(json.dumps(f" here is the otp: {phone_number+10}", indent=2))
    
except requests.exceptions.RequestException as e:
    error_message = str(e)
    try:
        error_message = f"{error_message}: {json.dumps("something went wrg")}"
    except:
        pass
    
    print(json.dumps({"error": error_message}))
    sys.exit(1)
