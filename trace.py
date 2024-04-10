import os 
os.environ["VERBOSE"] = "1"
# os.environ["NUM_PROCS"] = "1"

from lean_dojo import LeanGitRepo, trace

# repo = LeanGitRepo("https://github.com/yangky11/lean4-example", "1ac33f05939fdfa0a0e50a2fa65125a6474c07c0")
# repo = LeanGitRepo(url='https://github.com/leanprover/std4', commit='b1ebd72c5d262ea10a33ea582525925df874ad1e')
# repo = LeanGitRepo(url='https://github.com/leanprover-community/ProofWidgets4', commit='f5b2b6ff817890d85ffd8ed47637e36fcac544a6')
# repo = LeanGitRepo(url='https://github.com/yangky11/miniF2F-lean4', commit='d4ec261d2b9b8844f4ebfad4253cf3f42519c098')
repo = LeanGitRepo(url='https://github.com/leanprover-community/aesop', commit='e4660fa21444bcfe5c70d37b09cc0517accd8ad7')
trace(repo, dst_dir="deleteme")
