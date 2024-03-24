import psutil
import time
import argparse

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
            lean_file = proc.info['cmdline'][-1].strip()
            if lean_file != 'ExtractData.lean':
                goals.append(lean_file)
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

def has_enough_memory(threshold):
    mem = psutil.virtual_memory()
    used_percent = mem.percent
    return used_percent < threshold

def main(args):
    """Main function to run the script."""
    while True:
        time.sleep(args.timeout)
        if has_enough_memory(args.memory_threshold):
            continue
        lean_processes = get_lean_processes()
        if not lean_processes:
            continue
        goals = get_leandojo_goals(lean_processes)
        print(goals)
        if len(goals) < args.n_ld:
            continue
        assert len(goals) == args.n_ld
        zombies = get_zombies(lean_processes, goals)
        kill_lean_processes(zombies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description.")
    parser.add_argument("-t", "--timeout", type=int, default=5, help="Timeout in seconds")
    parser.add_argument("-n", "--n_ld", type=int, help="Number of Lean instances")
    parser.add_argument("-m", "--memory_threshold", type=int, help="Memory threshold in percent")
    args = parser.parse_args()
    assert args.n_ld is not None and args.memory_threshold is not None
    main(args)
