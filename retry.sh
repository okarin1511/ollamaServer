#!/bin/bash
set -e

MAX_RETRIES=5
COUNT=0

while [ $COUNT -lt $MAX_RETRIES ]; do
  echo "Attempt $(($COUNT + 1)) to run ollama..."
  if ollama run llama3.2:3b; then
    echo "Command succeeded."
    break
  else
    echo "Command failed with error. Retrying..."
    COUNT=$(($COUNT + 1))
    sleep 5
  fi
done

if [ $COUNT -eq $MAX_RETRIES ]; then
  echo "Command failed after $MAX_RETRIES attempts."
  exit 1
fi
