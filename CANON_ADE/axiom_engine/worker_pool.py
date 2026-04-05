"""
worker_pool.py — RAM-throttled multiprocessing pool with 24 workers.
Monitors system RAM via psutil and pauses workers if usage > 85%.
"""
import os
import time
import psutil
import multiprocessing as mp
from multiprocessing import Queue, Event, Value
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import traceback

RAM_LIMIT_PERCENT = 85
RAM_CHECK_INTERVAL = 2.0  # seconds
MAX_WORKERS = min(24, os.cpu_count() or 4)

class RAMThrottledPool:
    """Process pool that pauses task submission when RAM > threshold."""
    
    def __init__(self, max_workers=MAX_WORKERS, ram_limit=RAM_LIMIT_PERCENT,
                 max_ram_gb=64.0):
        self.max_workers = max_workers
        self.ram_limit = ram_limit
        self.max_ram_gb = max_ram_gb
        self._shutdown = threading.Event()
        self._paused = threading.Event()
        self._active_count = Value('i', 0)
        self._throttle_count = Value('i', 0)
        self._monitor_thread = None
    
    def _check_ram(self):
        """Return True if RAM is under limit."""
        mem = psutil.virtual_memory()
        used_gb = mem.used / 1e9
        pct = mem.percent
        return pct < self.ram_limit and used_gb < self.max_ram_gb
    
    def _ram_monitor(self):
        """Background thread: set _paused event when RAM is high."""
        while not self._shutdown.is_set():
            if not self._check_ram():
                if not self._paused.is_set():
                    self._paused.set()
                    with self._throttle_count.get_lock():
                        self._throttle_count.value += 1
            else:
                if self._paused.is_set():
                    self._paused.clear()
            time.sleep(RAM_CHECK_INTERVAL)
    
    def start_monitor(self):
        self._monitor_thread = threading.Thread(target=self._ram_monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitor(self):
        self._shutdown.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def map_with_throttle(self, fn, tasks, progress_callback=None):
        """
        Execute fn(task) for each task, throttling on RAM.
        progress_callback(completed, total, result) called on each completion.
        Returns list of results.
        """
        self.start_monitor()
        results = []
        total = len(tasks)
        completed = 0
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                task_iter = iter(enumerate(tasks))
                submitted = 0
                max_in_flight = self.max_workers * 2
                
                def submit_next():
                    nonlocal submitted
                    try:
                        idx, task = next(task_iter)
                        f = executor.submit(fn, task)
                        futures[f] = idx
                        submitted += 1
                        return True
                    except StopIteration:
                        return False
                
                # Initial batch
                for _ in range(min(max_in_flight, total)):
                    if not submit_next():
                        break
                
                while futures:
                    done = set()
                    for f in list(futures.keys()):
                        if f.done():
                            done.add(f)
                    
                    if not done:
                        time.sleep(0.05)
                        continue
                    
                    for f in done:
                        idx = futures.pop(f)
                        try:
                            result = f.result()
                            results.append(result)
                        except Exception as e:
                            results.append({'error': str(e), 'traceback': traceback.format_exc()})
                        
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total, results[-1])
                        
                        # Submit more if RAM allows
                        if not self._paused.is_set():
                            submit_next()
                        
                    # If paused, wait
                    while self._paused.is_set() and not self._shutdown.is_set():
                        time.sleep(0.5)
                    
                    # Try to keep pipeline full
                    while len(futures) < max_in_flight and not self._paused.is_set():
                        if not submit_next():
                            break
        
        finally:
            self.stop_monitor()
        
        return results
    
    @property
    def throttle_count(self):
        return self._throttle_count.value
    
    @property 
    def ram_status(self):
        mem = psutil.virtual_memory()
        return {
            'used_gb': mem.used / 1e9,
            'total_gb': mem.total / 1e9,
            'percent': mem.percent,
            'throttled': self._paused.is_set(),
        }
