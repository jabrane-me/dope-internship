import re
import sys

def extract_accuracy(log):
    match = re.search(r'auc\s(\d+\.\d+)', log)
    if match:
        return float(match.group(1))
    else:
        return None

if __name__ == "__main__":
    log = sys.stdin.read()
    accuracy = extract_accuracy(log)
    if accuracy is not None:
        print(accuracy)
    else:
        print("Accuracy not found")
