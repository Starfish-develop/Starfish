#!/bin/bash

echo "Opening website"
chromium http://localhost:8000/

echo "Starting server"
python -m http.server 8000
