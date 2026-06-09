import re

filepath = r"c:\Users\HP\James T\ML\body-fat-prediction\streamlit_app.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Replace all 1px solid borders with 2px solid borders, and increase opacity if it's white or general colors
content = re.sub(r'border:\s*1px solid', 'border: 2px solid', content)
content = re.sub(r'border-right:\s*1px solid', 'border-right: 2px solid', content)
content = re.sub(r'border-top:\s*1px solid', 'border-top: 2px solid', content)

# Make white borders more visible by replacing .05 or .08 with .15
content = re.sub(r'rgba\(255,\s*255,\s*255,\s*\.0[5-8]\)', 'rgba(255,255,255,.15)', content)
content = re.sub(r'rgba\(255,255,255,0\.0[5-8]\)', 'rgba(255,255,255,0.15)', content)

# A few specific borders like the red/blue/green ones:
content = re.sub(r'rgba\(239,68,68,0\.3\)', 'rgba(239,68,68,0.4)', content)
content = re.sub(r'rgba\(59,130,246,0\.3\)', 'rgba(59,130,246,0.4)', content)
content = re.sub(r'rgba\(34,197,94,0\.3\)', 'rgba(34,197,94,0.4)', content)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated borders.")
