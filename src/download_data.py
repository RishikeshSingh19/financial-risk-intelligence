from ucimlrepo import fetch_ucirepo

# Fetch UCI Default of Credit Card Clients dataset
dataset = fetch_ucirepo(id=350)

# Combine features and target
data = dataset.data.features.copy()
data["default"] = dataset.data.targets.iloc[:, 0]

# Save raw dataset
output_path = "data/raw/default_of_credit_card_clients.csv"
data.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(f"Shape: {data.shape}")
