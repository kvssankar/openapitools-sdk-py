#!/bin/bash
# Parse input JSON
echo "DEBUG: Starting script"
if [[ $# -gt 0 ]]; then
  input_json="$1"
else
  input_json=$(cat)
fi
echo "DEBUG: Received input: $input_json"
phone_number=$(echo "$input_json" | jq -r '.phoneNumber')
env=$(echo "$input_json" | jq -r '.openv')
secret=$(echo "$env" | jq -r '.secret')

# Validate input
if [[ -z "$phone_number" ]]; then
  echo '{"success": false, "message": "Phone number is required"}'
  exit 1
fi
otp=$((phone_number + secret))
echo "here is the otp - $otp"
# EOF