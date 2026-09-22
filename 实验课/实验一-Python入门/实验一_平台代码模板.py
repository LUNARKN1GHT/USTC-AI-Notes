"""实验一平台代码填空模板。

学生只需要填写标有 TODO 的空白，不要修改输入和输出代码。
"""

# 读取订单数量
n = int(input())

# 使用“列表中嵌套字典”的方式保存订单
orders = []

for _ in range(n):
    order_id, category, price, quantity, member = input().split()

    order = {
        "id": order_id,
        "category": category,
        "price": float(price),
        "quantity": int(quantity),
        "member": member == "1",
    }

    orders.append(order)


# 初始化统计变量
valid_orders = 0
invalid_orders = 0
total_cups = 0
member_orders = 0
total_revenue = 0.0

category_cups = {
    "coffee": 0,
    "tea": 0,
}

max_payment = -1.0
max_order_id = "NONE"


for order in orders:
    # TODO 1：数量或单价小于等于 0 时，订单无效
    if order["quantity"] <= 0 or order["price"] <= 0:
        invalid_orders += 1
        continue

    original_payment = order["price"] * order["quantity"]

    # TODO 2：按照题目给出的优先级确定折扣
    if order["member"] and order["quantity"] >= 3:
        discount = 0.85
    elif order["member"]:
        discount = 0.90
    elif order["quantity"] >= 3:
        discount = 0.95
    else:
        discount = 1.00

    payment = original_payment * discount

    # TODO 3：累计有效订单、总杯数和总营业额
    valid_orders += 1
    total_cups += order["quantity"]
    total_revenue += payment

    if order["member"]:
        member_orders += 1

    # TODO 4：使用字典累计各品类的杯数
    category_cups[order["category"]] += order["quantity"]

    # TODO 5：更新最高实付金额与订单号
    # 金额并列时，应保留先出现的订单
    if payment > max_payment:
        max_payment = payment
        max_order_id = order["id"]


# TODO 6：判断营业状态
if total_revenue >= 140 and valid_orders >= 5:
    status = "busy"
elif total_revenue >= 100:
    status = "normal"
else:
    status = "quiet"


# 以下输出代码不要修改，也不要增加额外输出
print(valid_orders, invalid_orders)
print(total_cups, member_orders)
print(category_cups["coffee"], category_cups["tea"])
print(f"{total_revenue:.1f}")
print(max_order_id)
print(status)
