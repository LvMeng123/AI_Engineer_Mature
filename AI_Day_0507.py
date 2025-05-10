import pandas as pd
import numpy as np

ts_shift_example = pd.Series([10, 20, 30, 40, 50],
                             index=pd.to_datetime(['2025-01-01', '2025-01-02',
                                                   '2025-01-03', '2025-01-04', '2025-01-05']))
print(f"\n--- .shift() 示例 ---")
print(f"原始序列:\n{ts_shift_example}\n")
print(f"向前移动1期 (滞后1期):\n{ts_shift_example.shift(1)}\n")
print(f"向后移动1期 (领先1期):\n{ts_shift_example.shift(-1)}\n")

# 计算日变化
daily_change = ts_shift_example - ts_shift_example.shift(1)
print(f"日变化:\n{daily_change}\n")