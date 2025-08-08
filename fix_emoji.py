#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# Read the file
file_path = r'c:\Users\Akash\OneDrive\Desktop\SMART INVENTORY CO-PILOT\src\app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic line with the correct emoji
# Use a pattern that matches the problematic character and replace the whole title
content = re.sub(
    r'st\.title\(".*?Simple Weather Guide"\)', 
    'st.title("🌤️ Simple Weather Guide")', 
    content
)

# Write back to file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed the weather page title emoji!")
