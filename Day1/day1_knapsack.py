"""
Day 01: Python 热身与环境综合实战
目标：
1. 使用 Python 高级特性 (List Comprehension, Type Hinting)
2. 结合 Numpy 生成数据
3. 使用 Gurobi 求解一个最简单的线性规划
4. 保存结果图表
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple  # 学习点：类型提示

# --- 技巧 1: 装饰器 (Decorator) ---
# 用于计算函数运行时间，这是工程化代码常用的技巧
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  [{func.__name__}] 耗时: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

# --- 技巧 2: 类型提示 (Type Hinting) ---
# 明确告诉读代码的人，n 是整数，返回的是一个元组
@timer_decorator
def generate_data(n: int) -> Tuple[np.ndarray, np.ndarray]:
    print(f"\n1️⃣  正在生成 {n} 条随机数据...")
    np.random.seed(42)  # 固定随机种子，保证结果可复现
    
    # --- 技巧 3: 列表推导式 vs Numpy ---
    # 虽然 Numpy 更快，但理解列表推导式对处理复杂逻辑很有用
    # items = [i for i in range(n)] 
    
    weights = np.random.randint(1, 10, n)  # 物品重量 1-10
    values = np.random.randint(10, 100, n) # 物品价值 10-100
    return weights, values

@timer_decorator
def solve_optimization(weights: np.ndarray, values: np.ndarray, capacity: int):
    print(f"\n2️⃣  正在使用 Gurobi 求解背包问题 (容量: {capacity})...")
    
    n = len(weights)
    
    # --- Gurobi 建模标准流程 ---
    try:
        # 1. 创建模型
        model = gp.Model("Day01_Knapsack")
        model.setParam('OutputFlag', 0)  # 0=不输出啰嗦的日志，1=输出
        
        # 2. 定义变量 (0/1 变量：选或不选)
        # using list comprehension to create variables efficiently
        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        
        # 3. 设定目标 (最大化总价值)
        # Gurobi 支持 quicksum，比 sum() 更快
        model.setObjective(gp.quicksum(x[i] * values[i] for i in range(n)), GRB.MAXIMIZE)
        
        # 4. 设定约束 (总重量 <= 容量)
        model.addConstr(gp.quicksum(x[i] * weights[i] for i in range(n)) <= capacity, "Capacity")
        
        # 5. 求解
        model.optimize()
        
        # 6. 输出结果
        if model.status == GRB.OPTIMAL:
            print(f"   ✅ 找到最优解! 总价值: {model.ObjVal:.1f}")
            
            # 获取被选中的物品索引
            selected_items = [i for i in range(n) if x[i].x > 0.5]
            print(f"   📦 选中的物品索引: {selected_items[:10]}... (只显示前10个)")
            return selected_items
        else:
            print("   ⚠️ 未找到最优解")
            return []
            
    except gp.GurobiError as e:
        print(f"   ❌ Gurobi Error: {e}")
        return []

@timer_decorator
def visualize_results(weights, values, selected_idx):
    print("\n3️⃣  正在绘图并保存...")
    
    # 简单的 Pandas 数据处理
    df = pd.DataFrame({
        'Weight': weights,
        'Value': values,
        'Selected': ['No'] * len(weights)
    })
    df.loc[selected_idx, 'Selected'] = 'Yes'
    
    # 画图
    plt.figure(figsize=(10, 6))
    
    # 画未选中的点
    plt.scatter(df[df['Selected']=='No']['Weight'], df[df['Selected']=='No']['Value'], 
                color='gray', alpha=0.5, label='Ignored')
    
    # 画选中的点
    plt.scatter(df[df['Selected']=='Yes']['Weight'], df[df['Selected']=='Yes']['Value'], 
                color='red', s=100, label='Selected (Optimal)')
    
    plt.title('Day 01: Knapsack Optimization Result')
    plt.xlabel('Weight')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    filename = 'day01_result.png'
    sctipt_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(sctipt_dir, filename)
    plt.savefig(filename)
    print(f"   🖼️  图片已保存为: {filename}")

if __name__ == "__main__":
    # 参数设置
    N_ITEMS = 50
    CAPACITY = 100
    
    # 执行流程
    w, v = generate_data(N_ITEMS)
    selected = solve_optimization(w, v, CAPACITY)
    visualize_results(w, v, selected)
    
    print("\n🎉 Day 01 学习任务完成！")