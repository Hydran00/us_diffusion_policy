import argparse
import json
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Ultrasound Diffusion Spline Policy")
    parser.add_argument(
        "--spline-policy-root", help="Path to the existing spline_policy repository"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("synthetic", help="Generate synthetic SOFTWARE TEST fixtures")
    demo.add_argument("--output", required=True)
    demo.add_argument("--config")
    demo.add_argument("--episodes", type=int, default=6)
    prep = sub.add_parser("prepare", help="Fit measured local spline targets and split groups")
    prep.add_argument("--raw", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--config")
    training = sub.add_parser("train")
    training.add_argument("--dataset", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--device", default="cpu")
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.add_argument("--device", default="cpu")
    quick = sub.add_parser(
        "smoke", help="CPU synthetic collection -> fitting -> training -> inference"
    )
    quick.add_argument("--output", required=True)
    convert = sub.add_parser("import-hdf5")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--mapping", required=True)
    extraction = sub.add_parser(
        "extract-anatomy", help="Extract Skin/Liver from the ultrasound Docker image"
    )
    extraction.add_argument("--output", default="data/assets/abdphantom")
    extraction.add_argument("--image", default="i4h_sim_build:ultrasound-simulator")
    anatomy = sub.add_parser(
        "anatomy", help="Estimate liver center/PCA and open the Open3D inspector"
    )
    anatomy.add_argument("--mesh-dir", default="data/assets/abdphantom")
    anatomy.add_argument("--output", required=True)
    anatomy.add_argument("--mesh-units", choices=["mm", "m"], default="mm")
    anatomy.add_argument(
        "--pose", help="JSON containing mesh_to_world (4x4, metres), exported from Isaac"
    )
    anatomy.add_argument("--display-voxel-m", type=float, default=0.003)
    anatomy.add_argument("--no-view", action="store_true")
    viewer = sub.add_parser("view-anatomy", help="Reopen saved anatomy in Open3D")
    viewer.add_argument("--input", required=True)
    viewer.add_argument("--screenshot")
    viewer.add_argument(
        "--capture-only",
        action="store_true",
        help="Capture one frame then close (requires a display)",
    )
    args = parser.parse_args()
    if args.command == "synthetic":
        from .demo import synthetic_episodes

        result = str(synthetic_episodes(args.output, Config.read(args.config), args.episodes))
    elif args.command == "prepare":
        from .data import prepare

        result = prepare(args.raw, args.output, Config.read(args.config), args.spline_policy_root)
    elif args.command == "train":
        from .train import train

        result = str(
            train(
                args.dataset,
                args.output,
                args.device,
                args.spline_policy_root,
                args.epochs,
                args.batch_size,
            )
        )
    elif args.command == "evaluate":
        from .train import evaluate

        result = evaluate(
            args.checkpoint,
            args.dataset,
            args.split,
            args.device,
            args.spline_policy_root,
        )
    elif args.command == "smoke":
        from .demo import smoke

        result = smoke(args.output, args.spline_policy_root)
    elif args.command == "import-hdf5":
        from .convert import import_hdf5

        result = str(
            import_hdf5(args.input, args.output, json.loads(Path(args.mapping).read_text()))
        )
    elif args.command == "extract-anatomy":
        from .anatomy_assets import extract_meshes

        result = str(extract_meshes(args.output, args.image))
    elif args.command == "anatomy":
        from .anatomy_viewer import prepare_anatomy, view_anatomy

        pose = json.loads(Path(args.pose).read_text())["mesh_to_world"] if args.pose else None
        result = prepare_anatomy(
            args.mesh_dir,
            args.output,
            mesh_to_world=pose,
            mesh_units=args.mesh_units,
            display_voxel_m=args.display_voxel_m,
        )
        print(json.dumps(result["liver_display_frame"], indent=2), flush=True)
        if not args.no_view:
            view_anatomy(args.output)
        result = {"landmarks": str(Path(args.output) / "landmarks.json")}
    elif args.command == "view-anatomy":
        from .anatomy_viewer import view_anatomy

        view_anatomy(args.input, screenshot=args.screenshot, close_after_capture=args.capture_only)
        result = {"input": args.input}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
