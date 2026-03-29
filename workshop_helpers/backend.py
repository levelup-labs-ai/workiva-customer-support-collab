import asyncio
import concurrent.futures

from agents import Agent, Runner, function_tool

RETURN_POLICY_DAYS = 30

ORDER_DB = {
    "ORD_9901": {"customer_id": "CUST_4821", "product": "SoundWave Pro Bluetooth Speaker", "date": "2024-03-08", "total": 79.99, "status": "delivered", "days_ago": 14},
    "ORD_8842": {"customer_id": "CUST_3307", "product": "ErgoRise Adjustable Laptop Stand", "date": "2024-03-18", "total": 54.99, "status": "delivered", "days_ago": 4},
    "ORD_7753": {"customer_id": "CUST_6614", "product": "Ceramic Pour-Over Coffee Set", "date": "2024-03-20", "total": 44.99, "status": "processing", "days_ago": 2, "flags": ["duplicate_charge"]},
    "ORD_6630": {"customer_id": "CUST_1145", "product": "Yoga Mat Bundle", "date": "2024-03-14", "total": 89.00, "status": "in_transit", "days_ago": 8, "tracking": "TRK_882991", "estimated_delivery": "2024-03-24", "delay_reason": "weather hold at regional hub"},
    "ORD_5517": {"customer_id": "CUST_9920", "product": "Stainless Steel Water Bottle 3-Pack", "date": "2024-03-22", "total": 38.50, "status": "processing", "days_ago": 0, "dispatch_window_minutes": 90},
    "ORD_4410": {"customer_id": "CUST_6623", "product": "Urban Commuter Backpack", "date": "2024-03-01", "total": 94.99, "status": "delivered", "days_ago": 21, "warranty_days": 90, "defect_confirmed": True},
    "ORD_3318": {"customer_id": "CUST_5540", "product": "Classic Canvas Tote - Navy Blue", "date": "2024-03-17", "total": 32.00, "status": "delivered", "days_ago": 5, "fulfillment_error": "wrong_colour_shipped", "correct_variant_in_stock": True},
    "ORD_2209": {"customer_id": "CUST_7765", "product": "SoundPro 400 Over-Ear Headphones", "date": "2024-03-08", "total": 119.99, "status": "delivered", "days_ago": 14},
    "ORD_8814": {"customer_id": "CUST_6601", "product": "ErgoRise Pro Laptop Stand", "date": "2024-02-16", "total": 69.99, "status": "delivered", "days_ago": 35},
    "ORD_7701": {"customer_id": "CUST_9934", "product": "Smart Home Starter Kit", "date": "2024-03-05", "total": 149.99, "status": "lost_in_transit", "days_ago": 17, "carrier_verdict": "confirmed_lost", "refund_due": True},
}

CUSTOMER_DB = {
    "CUST_4821": {"name": "Maya Patel", "status": "active", "orders": ["ORD_9901"]},
    "CUST_3307": {"name": "David Chen", "status": "active", "orders": ["ORD_8842"]},
    "CUST_6614": {"name": "Sophie Williams", "status": "active", "orders": ["ORD_7753"]},
    "CUST_2290": {"name": "James Okafor", "status": "active", "orders": [], "subscription": {"plan": "Premium Monthly", "price": 12.99, "enrolled": "2024-02-01", "last_billed": "2024-03-01", "auto_converted_from_trial": True}},
    "CUST_5503": {"name": "Priya Sharma", "status": "locked", "lock_reason": "5 failed login attempts at 09:14", "orders": []},
    "CUST_8871": {"name": "Tom Reeves", "status": "active", "orders": [], "email": "t.reeves_home@email.com", "last_reset_sent": "2024-03-22T10:42:00"},
    "CUST_1145": {"name": "Aisha Nguyen", "status": "active", "orders": ["ORD_6630"]},
    "CUST_9920": {"name": "Ben Hartley", "status": "active", "orders": ["ORD_5517"]},
    "CUST_3388": {"name": "Clara Johansson", "status": "active", "orders": [], "app_version": "4.1.0", "device": "Android 13", "latest_app_version": "4.2.1", "release_notes": "4.2.1 (2024-03-15): fixes order history crash on Android 13"},
    "CUST_7712": {"name": "Ryan Foster", "status": "active", "orders": [], "first_purchase_date": "2023-11-10", "promo_notes": "WELCOME15 is new-customer only - not eligible"},
    "CUST_4409": {"name": "Nina Kowalski", "status": "active", "orders": []},
    "CUST_6623": {"name": "Lena Fischer", "status": "active", "orders": ["ORD_4410"]},
    "CUST_8830": {"name": "Omar Hassan", "status": "active", "orders": [], "subscription": {"plan": "Premium Monthly", "price": 14.99, "next_billing": "2024-04-01", "cancellation_fee": 0}},
    "CUST_2271": {"name": "Fatima Al-Rashid", "status": "active", "orders": [], "subscription": {"plan": "Basic", "price": 9.00, "billing_day": 10, "upgrade_target": "Premium", "upgrade_price": 24.00, "prorate_today": 15.00}},
    "CUST_5540": {"name": "Jack Morrison", "status": "active", "orders": ["ORD_3318"]},
    "CUST_7765": {"name": "Sara Kim", "status": "active", "orders": ["ORD_2209"]},
    "CUST_3392": {"name": "Mark Davies", "status": "active", "orders": ["ORD_1101", "ORD_0982"], "expired_orders": ["ORD_0874"]},
    "CUST_6601": {"name": "Elena Vasquez", "status": "active", "orders": ["ORD_8814"]},
    "CUST_9934": {"name": "Carlos Mendez", "status": "active", "orders": ["ORD_7701"], "contact_history": [{"date": "2024-03-12", "note": "Advised to wait for delivery"}, {"date": "2024-03-16", "note": "Carrier investigation opened"}, {"date": "2024-03-19", "note": "Asked for update, no resolution"}]},
    "CUST_1187": {"name": "Aiko Tanaka", "status": "active", "orders": [], "subscription": {"plan": "Basic", "price": 9.00, "last_charge": 24.00, "overcharge_amount": 15.00, "billing_error": True}, "app_version": "4.0.9", "device": "Android", "latest_app_version": "4.2.1", "release_notes": "4.2.1 (2024-03-15): fixes random logout bug"},
}

PRODUCT_DB = {
    "QuickCharge 15W Wireless Pad": {
        "compatible_with": ["iPhone 8 and later", "all Android Qi devices"],
        "magsafe_compatible": True,
        "max_wattage_magsafe": 15,
        "max_wattage_qi": 7.5,
    }
}


def snapshot_backend() -> dict:
    return {
        "order_count": len(ORDER_DB),
        "customer_count": len(CUSTOMER_DB),
        "product_count": len(PRODUCT_DB),
    }


@function_tool
def get_customer_profile(customer_id: str) -> dict:
    """Get the full customer profile for the authenticated customer."""
    profile = CUSTOMER_DB.get(customer_id)
    if not profile:
        return {"error": f"No customer found: {customer_id}"}

    result = dict(profile)
    result["order_summaries"] = [
        {
            "order_id": order_id,
            "product": ORDER_DB.get(order_id, {}).get("product"),
            "date": ORDER_DB.get(order_id, {}).get("date"),
            "total": ORDER_DB.get(order_id, {}).get("order_total", ORDER_DB.get(order_id, {}).get("total")),
            "status": ORDER_DB.get(order_id, {}).get("status"),
            "days_ago": ORDER_DB.get(order_id, {}).get("days_ago"),
        }
        for order_id in profile.get("orders", [])
    ]
    return result


@function_tool
def get_order_details(order_id: str) -> dict:
    """Get the full order record for refunds, shipping issues, warranties, and fulfillment problems."""
    order = ORDER_DB.get(order_id)
    if not order:
        return {"error": f"Order not found: {order_id}"}
    return {"order_id": order_id, **order}


@function_tool
def check_return_eligibility(order_id: str) -> dict:
    """Check whether an order is within the 30-day return window."""
    order = ORDER_DB.get(order_id)
    if not order:
        return {"error": f"Order not found: {order_id}"}

    days_since_order = order.get("days_ago", 0)
    eligible = days_since_order <= RETURN_POLICY_DAYS
    return {
        "order_id": order_id,
        "product": order.get("product"),
        "order_total": order.get("total"),
        "days_since_order": days_since_order,
        "return_window": RETURN_POLICY_DAYS,
        "eligible": eligible,
        "days_over_window": max(0, days_since_order - RETURN_POLICY_DAYS),
        "recommendation": (
            "Approve return."
            if eligible
            else f"Decline - {days_since_order - RETURN_POLICY_DAYS} day(s) outside window. Offer supervisor escalation."
        ),
    }


@function_tool
def get_product_info(product_name: str) -> dict:
    """Look up product specifications and compatibility information."""
    info = PRODUCT_DB.get(product_name)
    if not info:
        return {"note": f"No detailed specs on file for: {product_name}. Use general knowledge."}
    return info


@function_tool
def take_action(customer_id: str, action: str, details: str) -> dict:
    """Perform a customer support action on behalf of the customer."""
    action_results = {
        "issue_refund": {"status": "refund_initiated", "eta_days": "3-5 business days"},
        "send_return_label": {"status": "label_sent", "eta_minutes": 5},
        "send_replacement": {"status": "replacement_dispatched", "eta_days": "1-2 business days"},
        "escalate": {"status": "escalated", "ticket_id": "ESC_00291", "eta": "supervisor review within 2 business hours"},
        "send_unlock_email": {"status": "unlock_email_sent", "eta_minutes": 1},
        "resend_reset_email": {"status": "reset_email_resent", "eta_minutes": 1},
        "cancel_subscription": {"status": "subscription_cancelled", "effective": "end of current period"},
        "apply_subscription_credit": {"status": "credit_applied", "eta_days": "3-5 business days"},
    }
    result = dict(action_results.get(action, {"status": "unknown_action", "action": action}))
    result["customer_id"] = customer_id
    result["details"] = details
    return result


TOOLS = [get_customer_profile, get_order_details, check_return_eligibility, get_product_info, take_action]


def default_support_agent_instructions(authenticated_customer_id: str) -> str:
    return (
        "You are a customer support agent for an online retail store. "
        f"The authenticated customer ID for this session is: {authenticated_customer_id}. "
        "Always call get_customer_profile first to understand the customer before responding. "
        "Use get_order_details for shipping, billing, warranty, and fulfillment issues tied to an order. "
        "Use check_return_eligibility before approving or declining any return. "
        "If the tools give you enough information to complete the request, you must take the action before replying. "
        "Use take_action for any refund, escalation, label, replacement, cancellation, or email. "
        "Do not merely promise an action in prose when a tool can perform it. "
        "Mention the confirmed action result in the response. "
        "Be empathetic and specific. Never invent figures or dates."
    )


def support_agent_instructions_template() -> str:
    return (
        "You are a customer support agent for an online retail store. "
        "The authenticated customer ID for this session is: {authenticated_customer_id}. "
        "Always call get_customer_profile first to understand the customer before responding. "
        "Use get_order_details for shipping, billing, warranty, and fulfillment issues tied to an order. "
        "Use check_return_eligibility before approving or declining any return. "
        "If the tools give you enough information to complete the request, you must take the action before replying. "
        "Use take_action for any refund, escalation, label, replacement, cancellation, or email. "
        "Do not merely promise an action in prose when a tool can perform it. "
        "Mention the confirmed action result in the response. "
        "Be empathetic and specific. Never invent figures or dates."
    )


def build_support_agent(
    authenticated_customer_id: str,
    model: str = "gpt-4o-mini",
    instructions: str | None = None,
) -> Agent:
    return Agent(
        name="Customer Support Agent",
        instructions=instructions or default_support_agent_instructions(authenticated_customer_id),
        tools=TOOLS,
        model=model,
    )


async def run_support_agent_async(
    customer_message: str,
    authenticated_customer_id: str,
    model: str = "gpt-4o-mini",
    instructions: str | None = None,
):
    agent = build_support_agent(
        authenticated_customer_id=authenticated_customer_id,
        model=model,
        instructions=instructions,
    )
    return await Runner.run(agent, customer_message)


def run_support_agent(
    customer_message: str,
    authenticated_customer_id: str,
    model: str = "gpt-4o-mini",
    instructions: str | None = None,
) -> dict:
    result = asyncio.run(
        run_support_agent_async(
            customer_message=customer_message,
            authenticated_customer_id=authenticated_customer_id,
            model=model,
            instructions=instructions,
        )
    )
    return {
        "output": result.final_output,
        "tool_calls": [
            raw.name
            for item in result.new_items
            for raw in [getattr(item, "raw_item", None)]
            if raw and hasattr(raw, "name")
        ],
    }


def run_support_agent_threadsafe(
    customer_message: str,
    authenticated_customer_id: str,
    model: str = "gpt-4o-mini",
    instructions: str | None = None,
) -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_support_agent, customer_message, authenticated_customer_id, model, instructions)
        return future.result()["output"]


def hydrate_backend_from_dataset(dataset: list[dict]) -> dict:
    for case in dataset:
        source_data = case["source_data"]
        customer_id = source_data.get("customer_id")
        order_id = source_data.get("order_id")

        if customer_id and customer_id not in CUSTOMER_DB:
            CUSTOMER_DB[customer_id] = {
                "name": source_data.get("customer_name", "Unknown"),
                "status": source_data.get("account_status", "active"),
                "orders": [order_id] if order_id else [],
            }
            for extra_field in [
                "subscription",
                "app_version",
                "device",
                "latest_app_version",
                "release_notes",
                "email",
                "last_reset_sent",
                "lock_reason",
                "first_purchase_date",
                "promo_notes",
                "contact_history",
            ]:
                if extra_field in source_data:
                    CUSTOMER_DB[customer_id][extra_field] = source_data[extra_field]

        if order_id and order_id not in ORDER_DB:
            ORDER_DB[order_id] = {
                "customer_id": customer_id,
                "product": source_data.get("product_name"),
                "date": source_data.get("order_date"),
                "total": source_data.get("order_total"),
                "status": source_data.get("order_status"),
                "days_ago": source_data.get("days_since_order"),
            }

    return snapshot_backend()
