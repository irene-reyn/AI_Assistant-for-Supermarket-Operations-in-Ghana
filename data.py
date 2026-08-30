from __future__ import annotations

import pandas as pd


PRODUCTS = [
    {"id": "P001", "name": "Rice", "category": "Grains", "price": 85.00, "stock": 24, "aisle": 1, "keywords": "rice, basmati, staple", "tags": "vegetarian, staple", "related": "Beans, Cooking Oil", "promotion": ""},
    {"id": "P002", "name": "Cooking Oil", "category": "Cooking", "price": 42.50, "stock": 18, "aisle": 1, "keywords": "oil, vegetable oil, frying", "tags": "vegetarian, staple", "related": "Rice, Tomato Paste", "promotion": ""},
    {"id": "P003", "name": "Beans", "category": "Grains", "price": 35.00, "stock": 30, "aisle": 1, "keywords": "beans, legumes, protein", "tags": "vegetarian, protein", "related": "Rice, Cooking Oil", "promotion": ""},
    {"id": "P004", "name": "Milk", "category": "Dairy", "price": 18.75, "stock": 12, "aisle": 2, "keywords": "milk, dairy, breakfast", "tags": "breakfast", "related": "Coffee, Sugar", "promotion": ""},
    {"id": "P005", "name": "Bread", "category": "Bakery", "price": 14.00, "stock": 9, "aisle": 2, "keywords": "bread, loaf, bakery", "tags": "breakfast", "related": "Milk, Margarine", "promotion": ""},
    {"id": "P006", "name": "Coffee", "category": "Beverages", "price": 28.00, "stock": 16, "aisle": 3, "keywords": "coffee, drink, breakfast", "tags": "breakfast", "related": "Milk, Sugar", "promotion": "5% off"},
    {"id": "P007", "name": "Sugar", "category": "Baking", "price": 16.50, "stock": 21, "aisle": 3, "keywords": "sugar, sweetener, baking", "tags": "baking", "related": "Coffee, Flour", "promotion": ""},
    {"id": "P008", "name": "Tomato Paste", "category": "Cooking", "price": 10.00, "stock": 0, "aisle": 4, "keywords": "tomato, paste, sauce", "tags": "vegetarian, cooking", "related": "Cooking Oil, Rice", "promotion": ""},
    {"id": "P009", "name": "Bottled Water", "category": "Beverages", "price": 6.00, "stock": 45, "aisle": 5, "keywords": "water, drink, bottle", "tags": "healthy", "related": "Bread, Coffee", "promotion": ""},
    {"id": "P010", "name": "Milo", "category": "Beverages", "price": 32.00, "stock": 11, "aisle": 3, "keywords": "milo, cocoa, chocolate drink", "tags": "breakfast", "related": "Milk, Sugar", "promotion": ""},
    {"id": "P011", "name": "Margarine", "category": "Dairy", "price": 22.00, "stock": 8, "aisle": 2, "keywords": "margarine, spread", "tags": "breakfast", "related": "Bread, Milk", "promotion": ""},
    {"id": "P012", "name": "Flour", "category": "Baking", "price": 25.00, "stock": 14, "aisle": 4, "keywords": "flour, baking, pastry", "tags": "baking, vegetarian", "related": "Sugar, Milk", "promotion": ""},
    {"id": "P013", "name": "Eggs", "category": "Dairy", "price": 30.00, "stock": 20, "aisle": 2, "keywords": "eggs, protein, breakfast", "tags": "breakfast, protein", "related": "Bread, Milk", "promotion": ""},
    {"id": "P014", "name": "Banana", "category": "Produce", "price": 12.00, "stock": 25, "aisle": 5, "keywords": "banana, fruit, fresh", "tags": "vegetarian, healthy, breakfast", "related": "Milk, Bread", "promotion": ""},
    {"id": "P015", "name": "Apples", "category": "Produce", "price": 24.00, "stock": 17, "aisle": 5, "keywords": "apple, apples, fruit, fresh", "tags": "vegetarian, healthy", "related": "Bottled Water, Bread", "promotion": ""},
    {"id": "P016", "name": "Toothpaste", "category": "Household", "price": 21.00, "stock": 10, "aisle": 6, "keywords": "toothpaste, personal care", "tags": "household", "related": "Bath Soap", "promotion": ""},
    {"id": "P017", "name": "Bath Soap", "category": "Household", "price": 13.00, "stock": 27, "aisle": 6, "keywords": "soap, bath, personal care", "tags": "household", "related": "Toothpaste", "promotion": ""},
    {"id": "P018", "name": "Chicken", "category": "Meat", "price": 65.00, "stock": 7, "aisle": 6, "keywords": "chicken, meat, protein", "tags": "protein", "related": "Cooking Oil, Rice", "promotion": ""},
    {"id": "P019", "name": "Peanut Butter", "category": "Spreads", "price": 38.00, "stock": 6, "aisle": 2, "keywords": "peanut, butter, spread", "tags": "breakfast, protein", "related": "Bread, Milk", "promotion": ""},
    {"id": "P020", "name": "Corn Flakes", "category": "Breakfast", "price": 34.00, "stock": 13, "aisle": 3, "keywords": "cereal, corn flakes, breakfast", "tags": "breakfast, vegetarian", "related": "Milk, Banana", "promotion": ""},
]


SYNTHETIC_USERS = [
    {"user_id": "U001", "name": "Ama Mensah", "language": "English", "budget": 150, "diet": "Vegetarian", "preferred_category": "Grains"},
    {"user_id": "U002", "name": "Kofi Owusu", "language": "Twi", "budget": 100, "diet": "No restriction", "preferred_category": "Beverages"},
    {"user_id": "U003", "name": "Esi Boateng", "language": "English", "budget": 200, "diet": "Lower sugar", "preferred_category": "Produce"},
    {"user_id": "U004", "name": "Yaw Asare", "language": "Twi", "budget": 120, "diet": "No restriction", "preferred_category": "Cooking"},
]


def load_products() -> pd.DataFrame:
    """Return the synthetic supermarket catalogue as a DataFrame."""
    return pd.DataFrame(PRODUCTS).copy()


def load_users() -> pd.DataFrame:
    """Return synthetic user profiles for demonstration and testing."""
    return pd.DataFrame(SYNTHETIC_USERS).copy()
