"""实验一：校园咖啡店订单统计。

请把 notebook 中完成的程序同步到本文件。
文件名改为：学号_姓名_实验一.py
"""

student_name = "请填写姓名"
student_id = "请填写学号"

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
    # TODO 1：数量或单价不合法时，累计 invalid_orders，打印原因并 continue。

    original_payment = order["price"] * order["quantity"]

    # TODO 2：按“会员且不少于 3 杯 / 会员 / 不少于 3 杯 / 其他”
    # 的优先级，用 if / elif / else 设置 discount。
    discount = 1.0
    payment = original_payment * discount

    # TODO 3：累计有效订单、总杯数、营业额和会员订单数。

    # TODO 4：在 category_cups 中按品类累计杯数。

    # TODO 5：更新最高实付金额及对应订单号。

    print(
        f"{order['id']} {order['drink']} {order['quantity']} 杯，实付 {payment:.1f} 元"
    )

# TODO 6：用 if / elif / else 判断“繁忙 / 正常 / 清闲”。
status = "待判断"

print("-" * 30)
print(f"总杯数：{total_cups}")
print(f"有效订单数：{valid_orders}")
print(f"无效订单数：{invalid_orders}")
print(f"会员订单数：{member_orders}")
print(f"分类杯数：{category_cups}")
print(f"总营业额：{total_revenue:.1f} 元")
print(f"最高实付订单：{max_order_id}")
print(f"营业状态：{status}")
