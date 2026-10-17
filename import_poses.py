"""Add keyframes from another poses file into this repo's poses.json, after the ones already there.

Typical use (Rishi, after pulling this branch): bring your arm2 keyframes in so "arm, then arm2" runs
Aaron's arm sequence first and then yours.

    uv run python import_poses.py ~/my_old_poses.json            # imports the arm2 keyframes
    uv run python import_poses.py ~/my_old_poses.json --arm arm   # imports arm keyframes instead
    uv run python import_poses.py ~/my_old_poses.json --dry-run   # show what would change

The source can be the old flat list or the grouped format. Only keyframes for the chosen arm are taken.
A keyframe whose name already exists for that arm is skipped (use --replace to overwrite it in place).
poses.json is backed up to poses.json.bak before it is changed.
"""
import argparse
import shutil
import sys
from pathlib import Path

import poses_io


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="poses file to import from")
    ap.add_argument("--arm", default="arm2", choices=poses_io.ARMS, help="which arm's keyframes to import")
    ap.add_argument("--target", default="poses.json", help="poses file to add them to")
    ap.add_argument("--replace", action="store_true", help="overwrite keyframes with the same name")
    ap.add_argument("--dry-run", action="store_true", help="print the result without writing")
    args = ap.parse_args()

    source, target = Path(args.source), Path(args.target)
    if not source.exists():
        sys.exit(f"{source} not found")
    if source.resolve() == target.resolve():
        sys.exit("source and target are the same file")
    incoming = [p for p in poses_io.load(source) if p["arm"] == args.arm]
    if not incoming:
        sys.exit(f"no {args.arm} keyframes in {source}")
    poses = poses_io.load(target)

    added, replaced, skipped = [], [], []
    for kf in incoming:
        at = next((i for i, p in enumerate(poses) if p["arm"] == args.arm and p["name"] == kf["name"]), None)
        if at is None:
            poses.append(kf)            # after everything already there
            added.append(kf["name"])
        elif args.replace:
            poses[at] = kf
            replaced.append(kf["name"])
        else:
            skipped.append(kf["name"])

    print(f"{args.arm} keyframes from {source}: added {added or 'none'}, replaced {replaced or 'none'}, "
          f"skipped (name exists) {skipped or 'none'}")
    print("play order:", ", ".join(f"{p['arm']}/{p['name']}" for p in poses))
    if args.dry_run:
        print("(dry run: nothing written)")
        return
    if target.exists():
        shutil.copyfile(target, target.with_name(target.name + ".bak"))
    poses_io.save(target, poses)
    print(f"wrote {target} (backup: {target.name}.bak)")


if __name__ == "__main__":
    main()
