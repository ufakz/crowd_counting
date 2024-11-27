import time
import tracemalloc
import psutil
import pandas as pd
from functools import wraps

class PerformanceMonitor:
    def __init__(self):
        self.metrics = pd.DataFrame(columns=[
            'function',
            'image_name',
            'time_took',
            'cpu_cycles', 
            'memory_usage',
            'memory_peak'
        ])
    
    def add_metric(self, func_name, image_name, duration, cpu_used, memory_usage, peak_memory):
        new_row = pd.DataFrame([{
            'function': func_name,
            'image_name': image_name,
            'time_took': duration,
            'cpu_cycles': cpu_used,
            'memory_usage': f'{memory_usage} MB',
            'memory_peak': f'{peak_memory} MB'
        }])
        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)
    
    def get_metrics(self):
        return self.metrics

    def measure_performance(self, func, image_name):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            
            # Initial measures 
            cpu_before = psutil.cpu_times()
            tracemalloc.start()
            start_memory = process.memory_info().rss / 1048576
            start_time = time.perf_counter()

            result = func(*args, **kwargs)

            # Final metrics
            end_time = time.perf_counter()
            cpu_after = psutil.cpu_times()
            end_memory = process.memory_info().rss / 1048576
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            cpu_used = cpu_after.user - cpu_before.user
            memory_usage = end_memory - start_memory
            duration = end_time - start_time
            approx_peak = peak / 10**6
            
            self.add_metric(
                func.__name__, 
                image_name,
                duration,
                cpu_used,
                memory_usage,
                approx_peak
            )
            return result
        
        return wrapper