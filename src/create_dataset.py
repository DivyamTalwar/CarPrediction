"""
Generate a realistic CarDekho-style dataset for car price prediction.
Based on the actual Kaggle CarDekho dataset structure and distributions.
"""
import pandas as pd
import numpy as np
import os

def create_dataset(output_path, n_samples=8000, seed=42):
    np.random.seed(seed)

    # Car brands and their base prices (in lakhs)
    cars = {
        'Maruti': {'models': ['Swift', 'Alto', 'Wagon R', 'Baleno', 'Dzire', 'Vitara Brezza', 'Ciaz', 'Ertiga', 'S-Cross', 'Celerio'],
                   'base_price': (3.5, 12.0)},
        'Hyundai': {'models': ['i20', 'Grand i10', 'Creta', 'Verna', 'Venue', 'Tucson', 'Santro', 'Xcent', 'Elite i20', 'i10'],
                    'base_price': (4.0, 18.0)},
        'Honda': {'models': ['City', 'Amaze', 'Jazz', 'WR-V', 'CR-V', 'Civic', 'Brio'],
                  'base_price': (5.0, 25.0)},
        'Toyota': {'models': ['Fortuner', 'Innova', 'Corolla', 'Etios', 'Camry', 'Yaris', 'Innova Crysta', 'Etios Cross'],
                   'base_price': (5.5, 35.0)},
        'Ford': {'models': ['EcoSport', 'Figo', 'Endeavour', 'Aspire', 'Freestyle'],
                 'base_price': (4.5, 30.0)},
        'Volkswagen': {'models': ['Polo', 'Vento', 'Ameo', 'Tiguan', 'Jetta'],
                       'base_price': (5.0, 20.0)},
        'Tata': {'models': ['Nexon', 'Tiago', 'Harrier', 'Safari', 'Hexa', 'Tigor', 'Nano', 'Bolt'],
                 'base_price': (3.0, 22.0)},
        'Mahindra': {'models': ['XUV500', 'Scorpio', 'Bolero', 'Thar', 'XUV300', 'Marazzo', 'KUV100'],
                     'base_price': (5.0, 22.0)},
        'Chevrolet': {'models': ['Beat', 'Cruze', 'Enjoy', 'Sail', 'Spark'],
                      'base_price': (3.0, 15.0)},
        'Renault': {'models': ['Kwid', 'Duster', 'Triber', 'Captur'],
                    'base_price': (3.0, 12.0)},
        'BMW': {'models': ['3 Series', '5 Series', 'X1', 'X3', 'X5', '7 Series'],
                'base_price': (25.0, 65.0)},
        'Audi': {'models': ['A4', 'A6', 'Q3', 'Q5', 'Q7', 'A3'],
                 'base_price': (25.0, 60.0)},
        'Mercedes-Benz': {'models': ['C-Class', 'E-Class', 'GLA', 'GLC', 'GLE', 'S-Class'],
                          'base_price': (30.0, 80.0)},
        'Skoda': {'models': ['Rapid', 'Superb', 'Octavia', 'Kodiaq'],
                  'base_price': (7.0, 28.0)},
        'Nissan': {'models': ['Kicks', 'Micra', 'Terrano', 'Sunny', 'Magnite'],
                   'base_price': (4.0, 15.0)},
    }

    # Brand probability weights (Maruti & Hyundai dominate Indian market)
    brand_weights = {
        'Maruti': 0.22, 'Hyundai': 0.18, 'Honda': 0.08, 'Toyota': 0.08,
        'Ford': 0.06, 'Volkswagen': 0.04, 'Tata': 0.10, 'Mahindra': 0.08,
        'Chevrolet': 0.03, 'Renault': 0.04, 'BMW': 0.02, 'Audi': 0.02,
        'Mercedes-Benz': 0.01, 'Skoda': 0.02, 'Nissan': 0.02
    }

    brands = list(cars.keys())
    weights = [brand_weights[b] for b in brands]

    data = []
    for _ in range(n_samples):
        # Pick brand
        brand = np.random.choice(brands, p=weights)
        model = np.random.choice(cars[brand]['models'])
        car_name = f"{brand} {model}"

        # Year (2003-2024, weighted towards recent years)
        year = int(np.random.choice(range(2003, 2025), p=_year_weights()))

        # Present price (ex-showroom, in lakhs)
        low, high = cars[brand]['base_price']
        present_price = round(np.random.uniform(low, high), 2)

        # Fuel type
        fuel_probs = [0.55, 0.35, 0.10]  # Petrol, Diesel, CNG
        fuel_type = np.random.choice(['Petrol', 'Diesel', 'CNG'], p=fuel_probs)
        if brand in ['BMW', 'Audi', 'Mercedes-Benz']:
            fuel_type = np.random.choice(['Petrol', 'Diesel'], p=[0.4, 0.6])

        # Seller type
        seller_type = np.random.choice(['Dealer', 'Individual'], p=[0.55, 0.45])

        # Transmission
        if brand in ['BMW', 'Audi', 'Mercedes-Benz']:
            transmission = np.random.choice(['Manual', 'Automatic'], p=[0.1, 0.9])
        else:
            transmission = np.random.choice(['Manual', 'Automatic'], p=[0.72, 0.28])

        # Owner
        owner = np.random.choice([0, 1, 3], p=[0.60, 0.30, 0.10])

        # Kms driven (correlated with age)
        car_age = 2024 - year
        avg_km_per_year = np.random.uniform(8000, 18000)
        kms_driven = int(car_age * avg_km_per_year + np.random.normal(0, 5000))
        kms_driven = max(500, kms_driven)

        # Calculate selling price based on realistic depreciation
        depreciation_rate = 0.15  # 15% per year base
        if fuel_type == 'Diesel':
            depreciation_rate = 0.12
        if transmission == 'Automatic':
            depreciation_rate -= 0.01
        if owner > 0:
            depreciation_rate += 0.03 * owner
        if seller_type == 'Individual':
            depreciation_rate += 0.02

        # Depreciation factor
        dep_factor = (1 - depreciation_rate) ** car_age
        dep_factor = max(0.05, dep_factor)  # Floor at 5% of original price

        # Add some noise
        noise = np.random.normal(1.0, 0.08)
        selling_price = round(present_price * dep_factor * noise, 2)
        selling_price = max(0.15, selling_price)  # Minimum 15k INR

        # Km-based adjustment
        km_penalty = max(0, (kms_driven - 50000)) * 0.000002
        selling_price = round(selling_price * (1 - km_penalty), 2)
        selling_price = max(0.15, selling_price)

        data.append({
            'Car_Name': car_name,
            'Year': year,
            'Selling_Price': selling_price,
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Fuel_Type': fuel_type,
            'Seller_Type': seller_type,
            'Transmission': transmission,
            'Owner': owner,
        })

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset created: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Price range: {df['Selling_Price'].min():.2f} - {df['Selling_Price'].max():.2f} lakhs")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    return df


def _year_weights():
    """Weight recent years more heavily."""
    years = list(range(2003, 2025))
    w = np.array([1.0 + i * 0.5 for i in range(len(years))])
    return w / w.sum()


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = os.path.join(project_dir, "data", "raw", "car_data.csv")
    create_dataset(output)
