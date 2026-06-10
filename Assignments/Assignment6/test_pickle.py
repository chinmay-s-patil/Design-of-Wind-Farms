import cupy as cp
import pickle

a = cp.array([1.0, 2.0])
try:
    pickle.dumps(a)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
