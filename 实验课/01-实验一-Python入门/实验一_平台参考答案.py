"""实验一平台自动判题参考答案，供教师录入与验收使用。"""

n = int(input())
orders = []

for _ in range(n):
    order_id, category, price, quantity, member = input().split()
    orders.append(
        {
            "id": order_id,
            "category": category,
            "price": float(price),
            "quantity": int(quantity),
            "member": member == "1",
        }
    )

valid_orders = 0
invalid_orders = 0
total_cups = 0
member_orders = 0
total_revenue = 0.0
category_cups = {"coffee": 0, "tea": 0}
max_payment = -1.0
max_order_id = "NONE"

for order in orders:
    if order["quantity"] <= 0 or order["price"] <= 0:
        invalid_orders += 1
        continue

    original_payment = order["price"] * order["quantity"]

    if order["member"] and order["quantity"] >= 3:
        discount = 0.85
    elif order["member"]:
        discount = 0.90
    elif order["quantity"] >= 3:
        discount = 0.95
    else:
        discount = 1.00

    payment = original_payment * discount
    valid_orders += 1
    total_cups += order["quantity"]
    total_revenue += payment

    if order["member"]:
        member_orders += 1

    category_cups[order["category"]] += order["quantity"]

    if payment > max_payment:
        max_payment = payment
        max_order_id = order["id"]

if total_revenue >= 140 and valid_orders >= 5:
    status = "busy"
elif total_revenue >= 100:
    status = "normal"
else:
    status = "quiet"

print(valid_orders, invalid_orders)
print(total_cups, member_orders)
print(category_cups["coffee"], category_cups["tea"])
print(f"{total_revenue:.1f}")
print(max_order_id)
print(status)
