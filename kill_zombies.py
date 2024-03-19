import psutil
import time

TIMEOUT = 5
# how many Lean instances are supposed to be
N_LD = 2
# threshold after that we kill zombies to clear RAM (in percent)
MEMORY_THRESHOLD = 70

def get_lean_processes():
    lean_processes = []
    for proc in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        if proc.info['cmdline'] and 'lean --threads' in ' '.join(proc.info['cmdline']):
            lean_processes.append(proc)
    return lean_processes

def get_leandojo_goals(lean_processes):
    goals = []
    for proc in lean_processes:
        if 'lake env' in ' '.join(proc.info['cmdline']):
            goals.append(proc.info['cmdline'][-1].strip())
    return goals

def get_zombies(lean_processes, goals):
    zombies = []
    for proc in lean_processes:
        if proc.info['cmdline'][-1] not in goals:
            zombies.append(proc)
    return zombies

def kill_lean_processes(lean_processes):
    for proc in lean_processes:
        print(f"Killing process {proc.pid} - {proc.info['cmdline']}")
        proc.kill()

def has_enough_memory():
    mem = psutil.virtual_memory()
    used_percent = mem.percent
    return used_percent < MEMORY_THRESHOLD

def main():
    """Main function to run the script."""
    while True:
        time.sleep(TIMEOUT)
        if has_enough_memory():
            continue
        lean_processes = get_lean_processes()
        if not lean_processes:
            continue
        goals = get_leandojo_goals(lean_processes)
        print(goals)
        if len(goals) < N_LD:
            continue
        # assert len(goals) == N_LD
        zombies = get_zombies(lean_processes, goals)
        kill_lean_processes(zombies)

if __name__ == "__main__":
    main()
