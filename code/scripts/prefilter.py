import sys
import datetime
std_out = sys.stdout
log_file = open(str(snakemake.log),'a')
sys.stdout = log_file
print(f"[{datetime.datetime.now()}] Log of rule {snakemake.rule}")

