import torch
import sys

def main():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"MPS (Apple Silicon) available: True")
    else:
        print(f"MPS (Apple Silicon) available: False")
    
    # Create a simple tensor
    x = torch.rand(3, 3)
    print(f"Random tensor:\n{x}")
    
    # Try a simple operation
    y = torch.nn.functional.softmax(x, dim=1)
    print(f"After softmax:\n{y}")
    
    print("PyTorch test completed successfully!")

if __name__ == "__main__":
    main() 