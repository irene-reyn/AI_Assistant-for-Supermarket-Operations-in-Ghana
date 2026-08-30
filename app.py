from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from data import load_products
from services import (
    LANGUAGE_PACKS,
    assistant_reply,
    cart_total,
    recommend_products,
    search_products,
)


st.set_page_config(
    page_title="Akwaaba Market Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_PROFILE: dict[str, Any] = {
    "name": "Guest Shopper",
    "budget": 150.0,
    "diet": "none",
    "category": "all",
    "language": "en",
    "purchase_history": ["Coffee", "Milk"],
}


@st.cache_data
def cached_products() -> pd.DataFrame:
    return load_products()


def initialise_state() -> None:
    defaults = {
        "products": cached_products().copy(),
        "profile": DEFAULT_PROFILE.copy(),
        "language": "en",
        "cart": {},
        "messages": [],
        "logs": [],
        "orders": [],
        "route": [],
        "route_product": None,
        "last_order": None,
        "sync_time": None,
        "presentation_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.messages:
        st.session_state.messages.append(
            {"role": "assistant", "content": LANGUAGE_PACKS["en"]["intro"]}
        )


def money(value: Any) -> str:
    return f"GH₵{float(value or 0):,.2f}"


def split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def product_by_id(product_id: str) -> pd.Series | None:
    products = st.session_state.products
    matches = products[products["id"] == product_id]
    return matches.iloc[0] if not matches.empty else None


def add_to_cart(product_id: str, quantity: int = 1) -> tuple[bool, str]:
    product = product_by_id(product_id)
    if product is None:
        return False, "Product was not found."

    available = int(product["stock"])
    if available <= 0:
        return False, f"{product['name']} is out of stock."

    current = int(st.session_state.cart.get(product_id, 0))
    new_quantity = min(current + max(1, quantity), available)
    st.session_state.cart[product_id] = new_quantity
    return True, f"{new_quantity} × {product['name']} is in your cart."


def remove_from_cart(product_id: str, amount: int = 1) -> None:
    current = int(st.session_state.cart.get(product_id, 0))
    new_quantity = current - amount
    if new_quantity <= 0:
        st.session_state.cart.pop(product_id, None)
    else:
        st.session_state.cart[product_id] = new_quantity


def cart_lines() -> list[dict[str, Any]]:
    lines = []
    for product_id, quantity in st.session_state.cart.items():
        product = product_by_id(product_id)
        if product is not None:
            lines.append({"product": product, "quantity": int(quantity)})
    return lines


def add_log(input_text: str, result: dict[str, Any], latency_ms: int) -> None:
    product = result.get("product")
    st.session_state.logs.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "language": st.session_state.language,
            "input": input_text,
            "intent": result.get("intent", "unknown"),
            "product": product["name"] if product is not None else "",
            "success": bool(result.get("success", False)),
            "latency_ms": latency_ms,
        }
    )


def process_prompt(prompt: str) -> None:
    prompt = str(prompt or "").strip()
    if not prompt:
        return

    started = time.perf_counter()
    st.session_state.messages.append({"role": "user", "content": prompt})
    result = assistant_reply(
        products=st.session_state.products,
        profile=st.session_state.profile,
        cart=st.session_state.cart,
        message=prompt,
        language=st.session_state.language,
    )

    if result.get("action") == "add" and result.get("product") is not None:
        ok, _ = add_to_cart(result["product"]["id"], int(result.get("quantity", 1)))
        result["success"] = ok
    elif result.get("action") == "checkout":
        st.session_state.checkout_open = True
    elif result.get("route"):
        st.session_state.route = result["route"]
        st.session_state.route_product = result.get("product")
    elif result.get("action") == "cart":
        st.session_state.checkout_open = False

    latency_ms = round((time.perf_counter() - started) * 1000)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["text"],
            "meta": f"{result.get('intent', 'unknown')} · {latency_ms} ms",
        }
    )
    add_log(prompt, result, latency_ms)


def complete_order(method: str, phone: str) -> None:
    lines = cart_lines()
    if not lines:
        st.warning("Your cart is empty.")
        return

    for line in lines:
        product = line["product"]
        if int(product["stock"]) < line["quantity"]:
            st.error(f"Not enough synthetic stock for {product['name']}.")
            return

    total = cart_total(st.session_state.products, st.session_state.cart)
    order_id = f"AK-{datetime.now().year}-{int(time.time()) % 1_000_000:06d}"

    for line in lines:
        mask = st.session_state.products["id"] == line["product"]["id"]
        st.session_state.products.loc[mask, "stock"] -= line["quantity"]
        product_name = line["product"]["name"]
        history = st.session_state.profile.setdefault("purchase_history", [])
        st.session_state.profile["purchase_history"] = list(dict.fromkeys(history + [product_name]))[-10:]

    order = {
        "order_id": order_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "customer": st.session_state.profile.get("name", "Guest Shopper"),
        "language": st.session_state.language,
        "method": method,
        "phone": phone if method == "MoMo" else "",
        "total": round(total, 2),
        "items": "; ".join(f"{line['quantity']} × {line['product']['name']}" for line in lines),
        "status": "Paid — simulation",
    }
    st.session_state.orders.append(order)
    st.session_state.last_order = order
    st.session_state.cart = {}
    st.session_state.checkout_open = False
    st.session_state.sync_time = datetime.now().isoformat(timespec="seconds")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Prototype controls")
        st.session_state.presentation_mode = st.toggle(
            "Presentation mode",
            value=st.session_state.presentation_mode,
            help="Shows a short guide for presenting the system.",
        )

        selected_language = st.selectbox(
            "Language",
            options=list(LANGUAGE_PACKS.keys()),
            format_func=lambda code: LANGUAGE_PACKS[code]["label"],
            index=list(LANGUAGE_PACKS.keys()).index(st.session_state.language),
        )
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.session_state.profile["language"] = selected_language
            st.session_state.messages.append(
                {"role": "assistant", "content": LANGUAGE_PACKS[selected_language]["intro"]}
            )

        st.divider()
        st.subheader("Synthetic customer profile")
        with st.form("profile_form"):
            name = st.text_input("Name", value=st.session_state.profile.get("name", "Guest Shopper"))
            budget = st.number_input("Budget in GH₵", min_value=0.0, value=float(st.session_state.profile.get("budget", 150)), step=10.0)
            diet_options = {"none": "No restriction", "vegetarian": "Vegetarian", "lowSugar": "Lower sugar"}
            diet = st.selectbox("Dietary preference", list(diet_options), format_func=lambda key: diet_options[key], index=list(diet_options).index(st.session_state.profile.get("diet", "none")))
            category_options = ["all", "Grains", "Cooking", "Dairy", "Beverages", "Bakery", "Produce", "Household", "Meat", "Breakfast", "Spreads"]
            current_category = st.session_state.profile.get("category", "all")
            category = st.selectbox("Preferred category", category_options, index=category_options.index(current_category) if current_category in category_options else 0)
            if st.form_submit_button("Save profile", type="primary"):
                st.session_state.profile.update({"name": name.strip() or "Guest Shopper", "budget": budget, "diet": diet, "category": category})
                st.success("Synthetic profile saved.")

        st.caption("All data is synthetic and stored in the current Streamlit session.")


def render_chat() -> None:
    st.subheader("Customer assistant")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("meta"):
                st.caption(message["meta"])

    quick_prompts = [
        "Where can I find rice?",
        "How much is cooking oil?",
        "Recommend something for breakfast",
        "Add milk to my cart",
        "Show my cart",
    ]
    prompt_columns = st.columns(len(quick_prompts))
    for column, quick_prompt in zip(prompt_columns, quick_prompts):
        if column.button(quick_prompt, key=f"quick_{quick_prompt}", use_container_width=True):
            process_prompt(quick_prompt)
            st.rerun()

    prompt = st.chat_input("Ask about products, prices, stock, recommendations, routes, or checkout")
    if prompt:
        process_prompt(prompt)
        st.rerun()

    if st.session_state.presentation_mode:
        st.info("Presentation order: profile → product question → route → recommendation → cart → simulated payment → admin evaluation.")


def render_catalogue() -> None:
    st.subheader("Product catalogue")
    search_query = st.text_input("Search products or categories", key="catalogue_search")
    category_options = ["All"] + sorted(st.session_state.products["category"].dropna().unique().tolist())
    selected_category = st.selectbox("Category filter", category_options, key="catalogue_category")
    stock_filter = st.selectbox("Stock filter", ["All", "In stock", "Out of stock"], key="catalogue_stock")

    if search_query:
        view = search_products(st.session_state.products, search_query, top_k=50)
    else:
        view = st.session_state.products.copy()
    if selected_category != "All":
        view = view[view["category"] == selected_category]
    if stock_filter == "In stock":
        view = view[view["stock"] > 0]
    elif stock_filter == "Out of stock":
        view = view[view["stock"] <= 0]

    if view.empty:
        st.info("No products match the current filters.")
        return

    for _, product in view.iterrows():
        columns = st.columns([2.5, 1.1, 1, 1, 1.2])
        columns[0].markdown(f"**{product['name']}**  \n{product['category']} · Aisle {int(product['aisle'])}")
        columns[1].markdown(f"**{money(product['price'])}**")
        columns[2].write(f"Stock: {int(product['stock'])}")
        if columns[3].button("Route", key=f"route_{product['id']}"):
            st.session_state.route_product = product
            st.session_state.route = ["Entrance", f"Aisle {int(product['aisle'])}", f"{product['name']} shelf", "Checkout"]
            st.rerun()
        if columns[4].button("Add", key=f"catalogue_add_{product['id']}", disabled=int(product["stock"]) <= 0):
            ok, message = add_to_cart(product["id"])
            if ok:
                st.toast(message)
            else:
                st.warning(message)
            st.rerun()


def render_cart() -> None:
    st.subheader("Shopping cart")
    lines = cart_lines()
    if not lines:
        st.info("Your cart is empty.")
    else:
        for line in lines:
            product = line["product"]
            columns = st.columns([2.4, 1, 0.7, 0.7, 0.7])
            columns[0].write(f"**{product['name']}**  \n{money(product['price'])} each")
            columns[1].write(f"Quantity: {line['quantity']}")
            if columns[2].button("−", key=f"minus_{product['id']}"):
                remove_from_cart(product["id"], 1)
                st.rerun()
            if columns[3].button("+", key=f"plus_{product['id']}", disabled=line["quantity"] >= int(product["stock"])):
                add_to_cart(product["id"], 1)
                st.rerun()
            if columns[4].button("×", key=f"remove_{product['id']}"):
                st.session_state.cart.pop(product["id"], None)
                st.rerun()
        st.metric("Cart total", money(cart_total(st.session_state.products, st.session_state.cart)))

    checkout_requested = st.button("Open checkout", type="primary", disabled=not bool(lines), use_container_width=True)
    if checkout_requested:
        st.session_state.checkout_open = True

    if st.session_state.get("checkout_open", False) and lines:
        st.divider()
        st.markdown("#### Simulated checkout")
        st.warning("This is a research simulation. No real payment is processed.")
        with st.form("checkout_form"):
            method = st.selectbox("Payment method", ["MoMo", "Cash"])
            phone = st.text_input("Synthetic MoMo number", value="0240000000", disabled=method != "MoMo")
            confirmed = st.form_submit_button("Confirm simulated payment", type="primary")
            if confirmed:
                complete_order(method, phone)
                st.rerun()

    if st.session_state.last_order:
        order = st.session_state.last_order
        st.success(f"Order {order['order_id']} completed in simulation. Total: {money(order['total'])}.")


def render_recommendations() -> None:
    st.subheader("Personalized recommendations")
    recommendations = recommend_products(
        st.session_state.products,
        st.session_state.profile,
        st.session_state.cart,
        context_product=st.session_state.route_product,
        top_k=5,
    )
    reason_parts = []
    profile = st.session_state.profile
    if profile.get("category") != "all":
        reason_parts.append(profile["category"])
    if profile.get("diet") == "vegetarian":
        reason_parts.append("vegetarian preference")
    if profile.get("diet") == "lowSugar":
        reason_parts.append("lower-sugar preference")
    if profile.get("budget"):
        reason_parts.append(f"budget {money(profile['budget'])}")
    reason = ", ".join(reason_parts) if reason_parts else "stock and product relationships"
    st.caption(f"Recommendation basis: {reason}.")

    if recommendations.empty:
        st.info("No suitable in-stock recommendations were found.")
        return
    for _, product in recommendations.iterrows():
        columns = st.columns([2.6, 1.2, 1])
        columns[0].write(f"**{product['name']}**  \n{product['category']} · Aisle {int(product['aisle'])}")
        columns[1].write(money(product["price"]))
        if columns[2].button("Add", key=f"recommend_add_{product['id']}"):
            add_to_cart(product["id"])
            st.rerun()


def render_route() -> None:
    st.subheader("Simulated store navigation")
    st.caption("This is the software navigation layer. It displays a route but does not control physical motors or sensors.")
    map_columns = st.columns(3)
    for aisle in range(1, 7):
        column = map_columns[(aisle - 1) % 3]
        aisle_products = st.session_state.products[st.session_state.products["aisle"] == aisle]["name"].tolist()
        label = f"**Aisle {aisle}**\n\n" + ("  \n".join(aisle_products) if aisle_products else "No products")
        if st.session_state.route_product is not None and int(st.session_state.route_product["aisle"]) == aisle:
            column.success(label)
        else:
            column.info(label)

    if st.session_state.route:
        product_name = st.session_state.route_product["name"] if st.session_state.route_product is not None else "selected product"
        st.markdown(f"**Route to {product_name}**")
        st.write(" → ".join(st.session_state.route))
        current_step = st.select_slider("Demonstration step", options=list(range(1, len(st.session_state.route) + 1)), format_func=lambda value: st.session_state.route[value - 1])
        st.progress(current_step / len(st.session_state.route))
        if st.button("Clear route"):
            st.session_state.route = []
            st.session_state.route_product = None
            st.rerun()


def render_customer_view() -> None:
    left, right = st.columns([1.2, 0.8])
    with left:
        render_chat()
        st.divider()
        render_catalogue()
    with right:
        render_cart()
        st.divider()
        render_recommendations()
        st.divider()
        render_route()


def render_admin_view() -> None:
    products = st.session_state.products
    logs = pd.DataFrame(st.session_state.logs)
    orders = pd.DataFrame(st.session_state.orders)
    successful = int(logs["success"].sum()) if not logs.empty else 0
    success_rate = successful / len(logs) if not logs.empty else 0

    metrics = st.columns(5)
    metrics[0].metric("Products", len(products))
    metrics[1].metric("Low stock", int((products["stock"] <= 5).sum()))
    metrics[2].metric("Interactions", len(logs))
    metrics[3].metric("Task success", f"{success_rate:.0%}")
    metrics[4].metric("Orders", len(orders))

    inventory_tab, order_tab, evaluation_tab = st.tabs(["Inventory", "Orders", "Evaluation"])

    with inventory_tab:
        st.subheader("Synthetic inventory management")
        st.caption("This table replaces live POS or ERP integration for the MVP. Changes immediately affect customer lookup and checkout.")
        editable_columns = ["id", "name", "category", "price", "stock", "aisle", "promotion"]
        editable = products[editable_columns].copy()
        edited = st.data_editor(
            editable,
            hide_index=True,
            num_rows="fixed",
            disabled=["id", "name", "category", "aisle"],
            column_config={
                "price": st.column_config.NumberColumn("Price GH₵", min_value=0, format="%.2f"),
                "stock": st.column_config.NumberColumn("Stock", min_value=0, step=1),
                "aisle": st.column_config.NumberColumn("Aisle", min_value=1, max_value=6, step=1),
            },
            key="inventory_editor",
        )
        if st.button("Apply inventory edits", type="primary"):
            for _, row in edited.iterrows():
                mask = st.session_state.products["id"] == row["id"]
                st.session_state.products.loc[mask, "price"] = float(row["price"])
                st.session_state.products.loc[mask, "stock"] = int(row["stock"])
                st.session_state.products.loc[mask, "promotion"] = str(row["promotion"])
            st.session_state.sync_time = datetime.now().isoformat(timespec="seconds")
            st.success("Synthetic inventory updated.")
            st.rerun()

        with st.expander("Add a synthetic product"):
            with st.form("add_product_form"):
                columns = st.columns(2)
                new_name = columns[0].text_input("Name")
                new_category = columns[1].text_input("Category")
                new_price = columns[0].number_input("Price GH₵", min_value=0.0, step=0.5)
                new_stock = columns[1].number_input("Stock", min_value=0, step=1)
                new_aisle = columns[0].number_input("Aisle", min_value=1, max_value=6, step=1)
                new_keywords = columns[1].text_input("Keywords, comma separated")
                new_related = st.text_input("Related products, comma separated")
                if st.form_submit_button("Add product"):
                    if not new_name or not new_category:
                        st.error("Product name and category are required.")
                    else:
                        new_id = f"P{len(st.session_state.products) + 1:03d}"
                        st.session_state.products.loc[len(st.session_state.products)] = {
                            "id": new_id,
                            "name": new_name,
                            "category": new_category,
                            "price": new_price,
                            "stock": new_stock,
                            "aisle": new_aisle,
                            "keywords": new_keywords,
                            "tags": "",
                            "related": new_related,
                            "promotion": "",
                        }
                        st.success(f"{new_name} added.")
                        st.rerun()

        if st.session_state.sync_time:
            st.caption(f"Last synthetic synchronization: {st.session_state.sync_time}")

    with order_tab:
        st.subheader("Simulated orders")
        if orders.empty:
            st.info("No simulated orders yet.")
        else:
            st.dataframe(orders, hide_index=True, use_container_width=True)

    with evaluation_tab:
        st.subheader("Evaluation evidence")
        st.caption("Logs support task success, intent accuracy checks, response latency, language comparison, and user testing.")
        if logs.empty:
            st.info("No interaction logs yet. Use the customer assistant first.")
        else:
            chart_columns = st.columns(2)
            with chart_columns[0]:
                st.write("Interactions by intent")
                st.bar_chart(logs["intent"].value_counts())
            with chart_columns[1]:
                st.write("Successful tasks by language")
                language_success = logs.groupby("language")["success"].mean().sort_values(ascending=False)
                st.bar_chart(language_success)
            st.dataframe(logs.sort_values("timestamp", ascending=False), hide_index=True, use_container_width=True)

        logs_csv = logs.to_csv(index=False).encode("utf-8") if not logs.empty else b""
        orders_csv = orders.to_csv(index=False).encode("utf-8") if not orders.empty else b""
        dataset_json = json.dumps({"products": products.to_dict(orient="records"), "orders": st.session_state.orders}, indent=2, default=str).encode("utf-8")
        download_columns = st.columns(3)
        download_columns[0].download_button("Download interaction CSV", logs_csv, "interaction_logs.csv", "text/csv", disabled=logs.empty)
        download_columns[1].download_button("Download orders CSV", orders_csv, "orders.csv", "text/csv", disabled=orders.empty)
        download_columns[2].download_button("Download synthetic dataset", dataset_json, "synthetic_dataset.json", "application/json")


def render_app() -> None:
    st.title("🛒 Akwaaba Market Assistant")
    st.caption("A shareable Streamlit prototype for multilingual supermarket assistance in Ghana")
    if st.session_state.presentation_mode:
        st.info("Presentation mode: demonstrate a customer task, route, recommendation, cart, simulated payment, then the evaluation dashboard.")

    render_sidebar()
    view = st.radio("Application view", ["Customer assistant", "Admin and evaluation"], horizontal=True, label_visibility="collapsed")
    if view == "Customer assistant":
        render_customer_view()
    else:
        render_admin_view()


initialise_state()
render_app()
