import subprocess
import psutil

def check_process_by_substring(substring):
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if not proc.info['cmdline']:
            continue
        if substring in ' '.join(proc.info['cmdline']):
            return True
    return False

file = 'nohup.out'

cmd = f'grep "proof!" {file} | wc -l && grep "Proving Theorem" {file} | wc -l'

# Running the command
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()

proved, total = map(int, out.decode().strip().split('\n'))
print(f"Started:\t{total}\nProved:\t\t{proved} ({round(proved/total*100, 1)}%)")

mem = psutil.virtual_memory()
print(f"\nUsed memory:\t{mem.percent}%")

status = "running" if check_process_by_substring("evaluate.py") else "finished"
print(f"\nStatus:\t\t{status}")
