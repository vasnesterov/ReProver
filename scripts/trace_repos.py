import argparse
import json
from glob import glob

from lean_dojo import LeanGitRepo, is_available_in_cache, trace
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()
    logger.info(args)

    url_commits = set()
    for path in glob(f"{args.data_path}/*/*/*.json"):
        data = json.load(open(path))
        for ex in data:
            url_commits.add((ex["url"], ex["commit"]))

    print(url_commits)

    repos = set()
    for url, commit in url_commits:
        repo = LeanGitRepo(url, commit)
        if not is_available_in_cache(repo):
            repos.add(repo)

    logger.info(f"Repos to trace: {repos}")

    for repo in repos:
        trace(repo)


if __name__ == "__main__":
    main()
