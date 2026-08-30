from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LANGUAGE_PACKS: dict[str, dict[str, Any]] = {
    "en": {
        "label": "English",
        "speech_code": "en-GH",
        "intro": "Hello! I can find products, check prices and stock, guide you to an aisle, recommend items, manage your cart, and simulate checkout.",
        "greeting": "Hello! How can I help you in the supermarket?",
        "not_found": "I could not match that product in the synthetic catalogue. Try rice, milk, coffee, cooking oil, bread, or banana.",
        "help": "Try: 'Where can I find rice?', 'How much is milk?', 'Is coffee available?', 'Recommend breakfast items', 'Add bread to my cart', or 'Checkout'.",
        "location": "{name} is in aisle {aisle}. The simulated route is shown below.",
        "price": "{name} costs GH₵{price}{promotion}. It is in aisle {aisle}.",
        "stock_in": "{name} is available. Stock: {stock}. Price: GH₵{price}.",
        "stock_out": "{name} is currently out of stock. It is listed in aisle {aisle}.",
        "added": "{quantity} × {name} was added to your cart.",
        "recommendation": "Based on {reason}, I recommend: {items}.",
        "recommendation_empty": "No suitable in-stock recommendation was found.",
        "cart_empty": "Your cart is empty. Ask me to add a product or use the catalogue.",
        "cart": "Your cart contains {count} product type(s), with a total of GH₵{total}.",
        "checkout": "Checkout is ready. Select a payment method and confirm the simulated payment.",
        "fallback": "I can help with product location, price, stock, recommendations, cart management, and simulated checkout.",
    },
    "tw": {
        "label": "Twi demonstration",
        "speech_code": "en-GH",
        "intro": "Akwaaba! Mebetumi aboa wo anya product, ahwɛ price ne stock, akyerɛ aisle, aka product akyerɛ wo, na ayɛ checkout demo. Twi nsɛm yi yɛ controlled demo; ma native speaker nhwɛ ansa na formal test.",
        "greeting": "Akwaaba! Ɛdeɛn na metumi aboa wo?",
        "not_found": "Mennyaa saa product no wɔ synthetic catalogue no mu. Sɔ rice, milk, coffee, cooking oil, bread, anaa banana bio.",
        "help": "Sɔ sɛ: 'Ɛhe na rice wɔ?', 'Milk boɔ yɛ sɛn?', 'Coffee wɔ hɔ?', 'Kamfo breakfast items', 'Fa bread kɔ me cart mu', anaa 'Checkout'.",
        "location": "{name} wɔ aisle {aisle}. Simulated route no wɔ ase ha.",
        "price": "{name} boɔ yɛ GH₵{price}{promotion}. Ɛwɔ aisle {aisle}.",
        "stock_in": "{name} wɔ hɔ. Stock: {stock}. Price: GH₵{price}.",
        "stock_out": "{name} nni stock seesei. Ɛwɔ aisle {aisle}.",
        "added": "{quantity} × {name} akɔ wo cart mu.",
        "recommendation": "Esiane {reason} nti, mekamfo: {items}.",
        "recommendation_empty": "Mennyaa recommendation a ɛwɔ stock mu.",
        "cart_empty": "Wo cart no yɛ hwee.",
        "cart": "Wo cart no wɔ product type {count}. Total no yɛ GH₵{total}.",
        "checkout": "Checkout abue. Paw payment method na si simulated payment no so dua.",
        "fallback": "Mebetumi aboa wɔ product location, price, stock, recommendation, cart, ne checkout demo ho.",
    },
    "ga": {
        "label": "Ga phrasebook",
        "speech_code": "en-GH",
        "intro": "Ga phrasebook mode is active. Validate Ga response phrases with a native speaker before formal language testing.",
    },
    "ee": {
        "label": "Ewe phrasebook",
        "speech_code": "en-GH",
        "intro": "Ewe phrasebook mode is active. Validate Ewe response phrases with a native speaker before formal language testing.",
    },
    "dag": {
        "label": "Dagbani phrasebook",
        "speech_code": "en-GH",
        "intro": "Dagbani phrasebook mode is active. Validate Dagbani response phrases with a native speaker before formal language testing.",
    },
}

# The first prototype uses English fallback templates for language packs whose
# verified translations have not yet been supplied. This avoids fabricating
# translations while keeping the language layer replaceable.
for _code in ("ga", "ee", "dag"):
    LANGUAGE_PACKS[_code] = {**LANGUAGE_PACKS["en"], **LANGUAGE_PACKS[_code]}


INTENT_TERMS: dict[str, dict[str, list[str]]] = {
    "en": {
        "greeting": ["hello", "hi", "hey"],
        "location": ["where", "find", "aisle", "location", "located", "route"],
        "price": ["price", "how much", "cost", "costs"],
        "stock": ["stock", "available", "in stock", "do you have"],
        "recommendation": ["recommend", "suggest", "something with", "breakfast", "alternative"],
        "add": ["add", "put", "include"],
        "cart": ["cart", "basket"],
        "checkout": ["checkout", "pay", "payment", "buy"],
    },
    "tw": {
        "greeting": ["akwaaba", "hello", "hi"],
        "location": ["ɛhe", "he na", "kyerɛ me", "aisle", "where", "route"],
        "price": ["boɔ", "bo", "sika", "price", "how much"],
        "stock": ["stock", "wɔ hɔ", "available"],
        "recommendation": ["kamfo", "susu", "recommend", "breakfast"],
        "add": ["fa", "ka ho", "cart mu", "add"],
        "cart": ["cart", "basket"],
        "checkout": ["checkout", "pay", "payment"],
    },
}
for _code in ("ga", "ee", "dag"):
    INTENT_TERMS[_code] = INTENT_TERMS["en"]


def normalize_text(value: Any) -> str:
    value = str(value or "").lower()
    value = unicodedata.normalize("NFD", value)
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def money(value: Any) -> str:
    return f"{float(value or 0):,.2f}"


def split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def product_search_text(row: pd.Series) -> str:
    values = [
        row.get("name", ""),
        row.get("category", ""),
        row.get("keywords", ""),
        row.get("tags", ""),
    ]
    return normalize_text(" ".join(str(value) for value in values))


def search_products(products: pd.DataFrame, query: str, top_k: int = 8) -> pd.DataFrame:
    """Search the synthetic catalogue using TF-IDF similarity."""
    if products.empty or not str(query or "").strip():
        return products.head(0).copy()

    documents = products.apply(product_search_text, axis=1).tolist()
    normalized_query = normalize_text(query)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(documents + [normalized_query])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()

    result = products.copy()
    result["_score"] = similarities
    result = result[result["_score"] > 0].sort_values("_score", ascending=False)
    return result.head(top_k).drop(columns=["_score"])


def detect_intent(message: str, language: str = "en") -> str:
    text = normalize_text(message)
    terms = INTENT_TERMS.get(language, INTENT_TERMS["en"])

    if any(text.startswith(normalize_text(word)) for word in terms["greeting"]):
        return "greeting"
    if any(normalize_text(word) in text for word in terms["checkout"]):
        return "checkout"
    if any(normalize_text(word) in text for word in terms["add"]):
        return "add"
    if any(normalize_text(word) in text for word in terms["cart"]):
        return "cart"
    if any(normalize_text(word) in text for word in terms["recommendation"]):
        return "recommendation"
    if any(normalize_text(word) in text for word in terms["location"]):
        return "location"
    if any(normalize_text(word) in text for word in terms["price"]):
        return "price"
    if any(normalize_text(word) in text for word in terms["stock"]):
        return "stock"
    if any(word in text for word in ("help", "what can", "boa", "bisa")):
        return "help"
    return "lookup"


def extract_quantity(message: str) -> int:
    match = re.search(r"\b(\d+)\b", str(message or ""))
    return max(1, int(match.group(1))) if match else 1


def find_product(products: pd.DataFrame, message: str) -> pd.Series | None:
    matches = search_products(products, message, top_k=1)
    if matches.empty:
        return None
    return matches.iloc[0]


def cart_total(products: pd.DataFrame, cart: dict[str, int]) -> float:
    total = 0.0
    for product_id, quantity in cart.items():
        match = products[products["id"] == product_id]
        if not match.empty:
            total += float(match.iloc[0]["price"]) * int(quantity)
    return total


def recommend_products(
    products: pd.DataFrame,
    profile: dict[str, Any],
    cart: dict[str, int],
    context_product: pd.Series | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return transparent, profile-aware recommendations."""
    if products.empty:
        return products.copy()

    cart_ids = set(cart.keys())
    related_names = set()
    if context_product is not None:
        related_names = {normalize_text(item) for item in split_values(context_product.get("related", ""))}

    history = {normalize_text(item) for item in profile.get("purchase_history", [])}
    preferred_category = str(profile.get("category", "all"))
    diet = str(profile.get("diet", "none"))
    budget = float(profile.get("budget", 0) or 0)

    scored = []
    for _, product in products.iterrows():
        if int(product["stock"]) <= 0 or product["id"] in cart_ids:
            continue

        score = 0.0
        tags = {normalize_text(item) for item in split_values(product.get("tags", ""))}
        name = normalize_text(product["name"])

        if preferred_category != "all" and product["category"] == preferred_category:
            score += 5
        if diet == "vegetarian" and "vegetarian" in tags:
            score += 5
        if diet == "vegetarian" and product["category"] == "Meat":
            score -= 20
        if diet == "lowSugar" and "healthy" in tags:
            score += 5
        if budget and float(product["price"]) <= budget:
            score += 3
        if name in history:
            score += 2
        if name in related_names:
            score += 8
        if str(product.get("promotion", "")).strip():
            score += 2

        score += max(0, 2 - float(product["price"]) / 100)
        scored.append((score, product))

    if not scored:
        return products.head(0).copy()

    scored.sort(key=lambda item: item[0], reverse=True)
    rows = [item[1] for item in scored[:top_k]]
    return pd.DataFrame(rows).reset_index(drop=True)


def build_route(product: pd.Series) -> list[str]:
    return [
        "Entrance",
        f"Aisle {int(product['aisle'])}",
        f"{product['name']} shelf",
        "Checkout",
    ]


def assistant_reply(
    products: pd.DataFrame,
    profile: dict[str, Any],
    cart: dict[str, int],
    message: str,
    language: str = "en",
) -> dict[str, Any]:
    """Resolve a user message into a transparent action and response."""
    pack = LANGUAGE_PACKS.get(language, LANGUAGE_PACKS["en"])
    intent = detect_intent(message, language)
    product = find_product(products, message)

    if intent == "greeting":
        return {"text": pack["greeting"], "intent": intent, "product": None, "success": True}
    if intent == "help":
        return {"text": pack["help"], "intent": intent, "product": None, "success": True}
    if intent == "checkout":
        return {"text": pack["checkout"], "intent": intent, "product": None, "success": True, "action": "checkout"}
    if intent == "cart":
        count = len(cart)
        total = money(cart_total(products, cart))
        text = pack["cart"] .format(count=count, total=total) if count else pack["cart_empty"]
        return {"text": text, "intent": intent, "product": None, "success": True, "action": "cart"}
    if intent == "recommendation":
        recommendations = recommend_products(products, profile, cart, product)
        items = ", ".join(f"{row['name']} (GH₵{money(row['price'])})" for _, row in recommendations.iterrows())
        text = pack["recommendation"].format(reason="your profile, cart, budget, and stock", items=items) if items else pack["recommendation_empty"]
        return {"text": text, "intent": intent, "product": product, "success": bool(items), "recommendations": recommendations}
    if intent == "add":
        if product is None:
            return {"text": pack["not_found"], "intent": intent, "product": None, "success": False}
        requested = extract_quantity(message)
        quantity = min(requested, int(product["stock"]))
        if quantity <= 0:
            text = pack["stock_out"].format(name=product["name"], aisle=int(product["aisle"]))
            return {"text": text, "intent": intent, "product": product, "success": False}
        return {"text": pack["added"].format(quantity=quantity, name=product["name"]), "intent": intent, "product": product, "success": True, "action": "add", "quantity": quantity}
    if product is None:
        return {"text": pack["not_found"], "intent": intent, "product": None, "success": False}

    promotion = f" ({product['promotion']})" if str(product.get("promotion", "")).strip() else ""
    if intent == "location":
        return {"text": pack["location"].format(name=product["name"], aisle=int(product["aisle"])), "intent": intent, "product": product, "success": True, "route": build_route(product)}
    if intent == "price":
        return {"text": pack["price"].format(name=product["name"], price=money(product["price"]), promotion=promotion, aisle=int(product["aisle"])), "intent": intent, "product": product, "success": True}
    if intent == "stock":
        if int(product["stock"]) > 0:
            text = pack["stock_in"].format(name=product["name"], stock=int(product["stock"]), price=money(product["price"]))
        else:
            text = pack["stock_out"].format(name=product["name"], aisle=int(product["aisle"]))
        return {"text": text, "intent": intent, "product": product, "success": True}

    text = f"{pack['price'].format(name=product['name'], price=money(product['price']), promotion=promotion, aisle=int(product['aisle']))} "
    text += pack["stock_in"].format(name=product["name"], stock=int(product["stock"]), price=money(product["price"])) if int(product["stock"]) > 0 else pack["stock_out"].format(name=product["name"], aisle=int(product["aisle"]))
    return {"text": text, "intent": "lookup", "product": product, "success": True}
