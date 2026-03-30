import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

collect_ignore_glob = ["*.pem", "*.prof"]
