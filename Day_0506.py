class Car:
    # 构造函数，初始化属性
    def __init__(self, brand, model, color):
        print(f"一辆新的 {color} {brand} {model} 被制造出来了！")
        self.brand = brand  # 品牌属性
        self.model = model  # 型号属性
        self.color = color  # 颜色属性
        self.is_running = False # 初始状态：未启动

    # 启动方法
    def start_engine(self):
        if not self.is_running:
            print(f"{self.brand} {self.model} 引擎启动！ Vroom!")
            self.is_running = True
        else:
            print("引擎已经在运行了。")

    # 停止方法
    def stop_engine(self):
        if self.is_running:
            print(f"{self.brand} {self.model} 引擎熄火。")
            self.is_running = False
        else:
            print("引擎本来就没启动。")

    # 鸣笛方法
    def honk(self):
        print(f"{self.brand} {self.model} 按喇叭：嘀嘀！")

    def __str__(self):
        status = '运行中' if self.is_running else '已熄火'
        return f"一辆{self.color}{self.brand}{self.model} ({status})"
# --- 类定义结束 ---

# 创建 Car 类的对象（实例）
my_car = Car("Toyota", "Camry", "银色")

# 调用对象的方法
print(my_car)