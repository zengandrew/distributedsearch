import os
import sys

# https://stackoverflow.com/a/45669280
class NoLogging:
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w') # standard printing
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w') # tqdm printing

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.stdout
        sys.stderr.close()
        sys.stderr = self.stderr
