import uuid

class MemoryBlock:
    def __init__(self, size, allocated_mode, type_info=None):
        self.address = uuid.uuid4()  # Simulated unique memory address.
        self.size = size
        self.allocated_mode = allocated_mode  # "MANUAL" or "AUTOMATIC"
        self.type_info = type_info
        self.marked = False  # Used by the GC to mark reachable objects.
    
    def __repr__(self):
        return f"MemoryBlock(addr={self.address}, size={self.size}, mode={self.allocated_mode})"

class ManualAllocator:
    def __init__(self):
        self.allocated_blocks = {}

    def alloc(self, size, type_info=None):
        block = MemoryBlock(size, "MANUAL", type_info)
        self.allocated_blocks[block.address] = block
        print(f"[ManualAllocator] Allocated {block}")
        return block

    def free(self, block):
        if block.address not in self.allocated_blocks:
            raise Exception(f"Double-free or invalid free attempted for block {block}")
        del self.allocated_blocks[block.address]
        print(f"[ManualAllocator] Freed {block}")

class GarbageCollector:
    def __init__(self):
        self.managed_blocks = {}  # Simulate a heap for automatic allocation

    def add_managed(self, block):
        self.managed_blocks[block.address] = block
        print(f"[GarbageCollector] Managed {block}")

    def mark(self):
        # In a real GC, this would traverse roots. Here we simulate marking.
        for block in self.managed_blocks.values():
            block.marked = True  # Simulate that all blocks are reachable.
        print("[GarbageCollector] Mark phase complete.")

    def sweep(self):
        to_free = [addr for addr, block in self.managed_blocks.items() if not block.marked]
        for addr in to_free:
            block = self.managed_blocks[addr]
            print(f"[GarbageCollector] Sweeping {block}")
            del self.managed_blocks[addr]
        print("[GarbageCollector] Sweep phase complete.")

    def collect(self):
        print("[GarbageCollector] Starting garbage collection cycle.")
        self.mark()
        self.sweep()
        # Reset marks for the next cycle.
        for block in self.managed_blocks.values():
            block.marked = False
        print("[GarbageCollector] GC cycle complete.")

class MemorySafetyEngine:
    def validate_allocation(self, block):
        # Placeholder: check if block size is valid, etc.
        if block.size <= 0:
            raise Exception(f"Invalid block size for {block}")
        return True

    def enforce_policy(self, block):
        # Placeholder: enforce custom policies, e.g., alignment, permissions, etc.
        print(f"[MemorySafetyEngine] Enforcing policies on {block}")
        return True

class MemoryManager:
    def __init__(self):
        self.manual_allocator = ManualAllocator()
        self.garbage_collector = GarbageCollector()
        self.safety_engine = MemorySafetyEngine()

    def alloc_manual(self, size, type_info=None):
        block = self.manual_allocator.alloc(size, type_info)
        self.safety_engine.validate_allocation(block)
        return block

    def free(self, block):
        # For manual allocations, free using the manual allocator.
        self.safety_engine.enforce_policy(block)
        self.manual_allocator.free(block)

    def alloc_automatic(self, size, type_info=None):
        block = MemoryBlock(size, "AUTOMATIC", type_info)
        self.safety_engine.validate_allocation(block)
        self.garbage_collector.add_managed(block)
        return block

    def run_gc(self):
        self.garbage_collector.collect()

# For testing purposes.
if __name__ == "__main__":
    mem_manager = MemoryManager()
    
    # Test manual allocation
    block1 = mem_manager.alloc_manual(64, type_info="int")
    mem_manager.free(block1)
    
    # Test automatic allocation and GC
    block2 = mem_manager.alloc_automatic(128, type_info="float")
    block3 = mem_manager.alloc_automatic(256, type_info="string")
    
    # Simulate that block3 is unreachable by manually unmarking it.
    block3.marked = False
    block2.marked = True  # Block2 remains reachable.
    
    mem_manager.run_gc()
    
    # Check remaining managed blocks.
    print("Remaining managed blocks:", mem_manager.garbage_collector.managed_blocks)
