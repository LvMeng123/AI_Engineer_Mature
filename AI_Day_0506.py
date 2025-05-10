import time
import functools # 导入 functools 模块

def timer(func):
    """一个简单的计时装饰器"""
    @functools.wraps(func) # 保留原函数的元信息(如名字、文档字符串)
    def wrapper(*args, **kwargs):
        # *args 接收所有位置参数 (组成元组)
        # **kwargs 接收所有关键字参数 (组成字典)
        start_time = time.perf_counter()
        print('代理函数开始执行')
        result = func(*args, **kwargs) # 调用原始函数
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"函数 '{func.__name__}' 执行耗时: {run_time:.6f} 秒")
        return result # 返回原始函数的执行结果
    return wrapper # 返回包装后的函数

# --- 装饰器定义结束 ---

@timer # 使用 @ 语法糖应用装饰器
def slow_function(delay):
    """一个模拟耗时的函数"""
    print(f"函数开始执行，将暂停 {delay} 秒...")
    time.sleep(delay)
    print("函数执行结束。")
    return "执行完毕"

@timer
def another_function(a, b=10):
    """另一个需要计时的函数"""
    print("另一个函数执行...")
    return a + b

# --- 调用被装饰的函数 ---
result1 = slow_function(2) # 传入位置参数
print(f"slow_function 返回结果: {result1}\n")

result2 = another_function(5, b=20) # 传入位置参数和关键字参数
print(f"another_function 返回结果: {result2}")