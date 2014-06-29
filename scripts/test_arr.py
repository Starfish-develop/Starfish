import argparse

parser = argparse.ArgumentParser(prog="test_arr.py", description="Identify Job Array variables")
parser.add_argument("--littlea")
parser.add_argument("--biga")
args = parser.parse_args()

print("a =", args.littlea)
print("A =", args.biga)
