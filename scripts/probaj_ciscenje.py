from ab_agent.agent.loading import load_csv, sanitize_text

# 1) Probaj funkciju za ciscenje direktno
print(repr(sanitize_text("plat\tform", 64)))
print(repr(sanitize_text("   previse    razmaka   ", 64)))
print(repr(sanitize_text("x" * 100, 10)))
print()

# 2) Probaj ucitavanje "prljavog" fajla
df = load_csv("data/prljav_test.csv")
print("Kolone:", list(df.columns))
print("Grupe :", list(df["variant"].unique()))