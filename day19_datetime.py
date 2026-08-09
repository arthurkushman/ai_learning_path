from datetime import datetime

today = datetime.today()
print(today)
print(today.year)
print(today.month)

start = datetime(2026, 1, 1, 0, 0, 1)
left = today - start
print(f"{left.days} days passed since the start of the year")

print(today.strftime("%A, %d %B %Y %H:%M"))