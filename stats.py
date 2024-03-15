import subprocess

file = 'nohup.out'

cmd = f'grep "proof!" {file} | wc -l && grep "Proving Theorem" {file} | wc -l'

# Running the command
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()

proved, total = map(int, out.decode().strip().split('\n'))
print(f"Total:\t{total}\nProved:\t{proved} ({round(proved/total*100, 1)}%)")
