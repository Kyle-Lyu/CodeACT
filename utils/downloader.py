import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
from huggingface_hub import hf_hub_download, snapshot_download 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", type=str, default=None,
    )
    parser.add_argument(
        "--local_dir", type=str, default=None,
    )
    parser.add_argument(
        "--filename", type=str, default=None,
    )
    parser.add_argument(
        "--resume_download", action="store_true",
    )
    parser.add_argument(
        "--ignore_patterns", nargs='+',
    )
    parser.add_argument(
        "--token", type=str, default=None,
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
    # exit()

    if args.filename:
        # Download a given file if it's not already present in the local cache.
        hf_hub_download(
            repo_id=args.repo_id, 
            filename=args.filename, 
            local_dir=args.local_dir, 
            local_dir_use_symlinks=False,
            token=args.token,
            cache_dir=args.cache_dir,
        )
    else:
        # Download a whole snapshot of a repo's files at the specified revision.
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            local_dir_use_symlinks=False,
            resume_download=args.resume_download,
            ignore_patterns=args.ignore_patterns,
            token=args.token,
            cache_dir=args.cache_dir,
        )
    