import sys
import numpy as np
import time
import os

def mb_to_bytes(mb_str):
    """
    Converts a string representing Megabytes (MB) into bytes.
    """
    try:
        value = int(mb_str.strip())
        if value <= 0:
            raise ValueError("Size must be a positive integer.")
        return value * 1024 * 1024
    except ValueError:
        raise ValueError(f"Input must be an integer number of Megabytes. Received: {mb_str}")


def allocate_memory(requested_mb_str):
    """
    Allocates a numpy array of the specified size in MB and keeps it in memory.
    """
    try:
        total_bytes = mb_to_bytes(requested_mb_str)
        requested_mb = int(requested_mb_str)
    except ValueError as e:
        print(f"Error parsing size: {e}")
        sys.exit(1)

    # Use float64 (8 bytes per element) for accurate calculation
    DTYPE_SIZE = 8
    num_elements = total_bytes // DTYPE_SIZE
    
    print(f"Memory Tester Initialized")
    print(f"Requested memory: {requested_mb} MB ({total_bytes} bytes)")
    print(f"Creating a NumPy array of {num_elements:,} elements (dtype=float64)...")

    # Allocate the memory by creating a large array of zeros
    try:
        # Using np.zeros forces the allocation to happen immediately
        arr = np.ones(num_elements, dtype=np.float64) 
        
        # Accessing the array helps ensure the memory is physically allocated
        temp = arr[0] 
        
        print(f"Successfully allocated array of size: {arr.nbytes / (1024**3):.3f} GB.")
        # Removed the system memory check as it often fails in containerized Condor environments
        print("Array allocated and held in memory. Sleeping for 60 seconds...")

        # Sleep long enough for the Condor or OOM Killer daemon to check memory usage
        time.sleep(60)

        print("Finished sleep cycle. Exiting successfully.")
        del arr

    except MemoryError:
        print(f"FAILED: Requested memory size {requested_mb} MB is too large for the environment.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python memory_tester.py <memory_size_in_MB>")
        print("Example: python memory_tester.py 512")
        print("Example: python memory_tester.py 2048 (for 2 GB)")
        sys.exit(1)
        
    requested_size = sys.argv[1]
    allocate_memory(requested_size)
