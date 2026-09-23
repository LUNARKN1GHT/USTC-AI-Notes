"""实验一参考答案，供助教验收与数据替换测试。"""

orders = [
    {
        "id": "A001",
        "category": "咖啡",
        "drink": "美式",
        "price": 12.0,
        "quantity": 2,
        "member": True,
    },
    {
        "id": "A002",
        "category": "咖啡",
        "drink": "拿铁",
        "price": 18.0,
        "quantity": 1,
        "member": False,
    },
    {
        "id": "A003",
        "category": "茶饮",
        "drink": "红茶",
        "price": 10.0,
        "quantity": 3,
        "member": True,
    },
    {
        "id": "A004",
        "category": "咖啡",
        "drink": "摩卡",
        "price": 20.0,
        "quantity": 2,
        "member": False,
    },
    {
        "id": "A005",
        "category": "咖啡",
        "drink": "美式",
        "price": 12.0,
        "quantity": 4,
        "member": True,
    },
    {
        "id": "A006",
        "category": "茶饮",
        "drink": "绿茶",
        "price": 9.0,
        "quantity": 0,
        "member": False,
    },
]

total_cups = 0
total_revenue = 0.0
member_orders = 0
valid_orders = 0
invalid_orders = 0
category_cups = {}
max_payment = -1.0
max_order_id = ""

for order in orders:
    if order["quantity"] <= 0 or order["price"] <= 0:
        invalid_orders += 1
        print(f"{order['id']} 数据无效，已跳过")
        continue

    original_payment = order["price"] * order["quantity"]
    if order["member"] and order["quantity"] >= 3:
        discount = 0.85
    elif order["member"]:
        discount = 0.9
    elif order["quantity"] >= 3:
        discount = 0.95
    else:
        discount = 1.0
    payment = original_payment * discount

    valid_orders += 1
    if order["member"]:
        member_orders += 1

    total_cups += order["quantity"]
    total_revenue += payment
    category = order["category"]
    if category not in category_cups:
        category_cups[category] = 0
    category_cups[category] += order["quantity"]

    if payment > max_payment:
        max_payment = payment
        max_order_id = order["id"]

    print(
        f"{order['id']} {order['drink']} "
        f"{order['quantity']} 杯，折扣 {discount:.2f}，实付 {payment:.1f} 元"
    )

if total_revenue >= 140 and valid_orders >= 5:
    status = "繁忙"
elif total_revenue >= 100:
    status = "正常"
else:
    status = "清闲"

print("-" * 30)
print(f"总杯数：{total_cups}")
print(f"有效订单数：{valid_orders}")
print(f"无效订单数：{invalid_orders}")
print(f"会员订单数：{member_orders}")
print(f"分类杯数：{category_cups}")
print(f"总营业额：{total_revenue:.1f} 元")
print(f"最高实付订单：{max_order_id}")
print(f"营业状态：{status}")
