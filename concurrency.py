import threading
import asyncio
from queue import Queue

# 1. Thread Manager
class ThreadManager:
    def __init__(self):
        self.threads = []

    def create_thread(self, target, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        thread = threading.Thread(target=target, args=args, kwargs=kwargs)
        self.threads.append(thread)
        thread.start()
        return thread

    def join_thread(self, thread):
        thread.join()

    def join_all(self):
        for thread in self.threads:
            thread.join()

# 2. Async Scheduler using asyncio
class AsyncScheduler:
    def __init__(self):
        self.loop = asyncio.get_event_loop()

    async def schedule_coroutine(self, coro):
        return await coro

    def run(self, coro):
        # Run a coroutine to completion.
        return self.loop.run_until_complete(coro)

# 3. Synchronization Primitives
class Mutex:
    def __init__(self):
        self._lock = threading.Lock()
    
    def lock(self):
        self._lock.acquire()
    
    def unlock(self):
        self._lock.release()

class Semaphore:
    def __init__(self, value=1):
        self._sem = threading.Semaphore(value)
    
    def wait(self):
        self._sem.acquire()
    
    def signal(self):
        self._sem.release()

class ConditionVariable:
    def __init__(self):
        self._cond = threading.Condition()
    
    def wait(self, mutex):
        # The mutex here is assumed to be a threading.Lock (or our Mutex wrapper's internal lock)
        with self._cond:
            self._cond.wait()
    
    def notify(self):
        with self._cond:
            self._cond.notify()
    
    def notify_all(self):
        with self._cond:
            self._cond.notify_all()

# Example usage and tests for the concurrency module.
if __name__ == "__main__":
    import time

    # Test ThreadManager
    def worker(name, delay):
        print(f"[Thread] {name} started")
        time.sleep(delay)
        print(f"[Thread] {name} finished")

    print("=== Testing ThreadManager ===")
    tm = ThreadManager()
    t1 = tm.create_thread(target=worker, args=("Thread-1", 1))
    t2 = tm.create_thread(target=worker, args=("Thread-2", 2))
    tm.join_all()
    print("All threads joined.\n")

    # Test AsyncScheduler
    print("=== Testing AsyncScheduler ===")
    async def async_worker(name, delay):
        print(f"[Async] {name} started")
        await asyncio.sleep(delay)
        print(f"[Async] {name} finished")
        return name

    scheduler = AsyncScheduler()
    result = scheduler.run(async_worker("Async-1", 1.5))
    print(f"Async result: {result}\n")

    # Test Synchronization Primitives: Mutex
    print("=== Testing Mutex ===")
    shared_resource = 0
    mutex = Mutex()

    def increment_resource():
        global shared_resource
        for _ in range(10000):
            mutex.lock()
            shared_resource += 1
            mutex.unlock()

    threads = []
    for i in range(10):
        t = threading.Thread(target=increment_resource)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print(f"Shared resource value: {shared_resource}\n")

    # Test Semaphore
    print("=== Testing Semaphore ===")
    sem = Semaphore(2)
    def sem_worker(id):
        print(f"[Semaphore] Worker {id} waiting")
        sem.wait()
        print(f"[Semaphore] Worker {id} acquired semaphore")
        time.sleep(1)
        print(f"[Semaphore] Worker {id} releasing semaphore")
        sem.signal()

    for i in range(4):
        tm.create_thread(target=sem_worker, args=(i,))
    tm.join_all()
    
    print("=== All tests completed ===")
