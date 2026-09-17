import argparse
import json
from pathlib import Path

from us_dp.config import Config


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
    prep.add_argument("--raw", "--input", dest="raw", required=True,
                      help="Raw NPZ directory, i4h run directory, or canonical i4h HDF5 file")
    prep.add_argument("--output", required=True)
    prep.add_argument("--config")
    training = sub.add_parser("train")
    training.add_argument("--dataset", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--device", default="cpu")
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--no-image-conditioning", action="store_true",
                          help="Train with state-only conditioning; persist mode in checkpoint")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--dataset", required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.add_argument("--device", default="cpu")
    comparison = sub.add_parser(
        "compare-image-conditioning",
        help="Evaluate an image-conditioned and a pose-only checkpoint on the same test set",
    )
    comparison.add_argument("--checkpoint-a", required=True)
    comparison.add_argument("--checkpoint-b", required=True)
    comparison.add_argument("--dataset",
                            help="Prepared dataset both checkpoints were trained on")
    comparison.add_argument("--split", choices=["validation", "test"], default="test")
    comparison.add_argument("--device", default="cpu")
    comparison.add_argument("--output", default="compare_image_conditioning.md",
                            help="Path for the Markdown comparison table")
    comparison.add_argument("--sim-episodes", type=int, default=0,
                            help="Run N complete closed-loop episodes per checkpoint in Isaac")
    comparison.add_argument("--sim-seed", type=int, default=42)
    comparison.add_argument("--skip-offline", action="store_true",
                            help="Skip window metrics and run simulator episodes directly")
    sensitivity = sub.add_parser(
        "image-sensitivity",
        help="Check whether an image-conditioned checkpoint actually uses the ultrasound image",
    )
    sensitivity.add_argument("--checkpoint", required=True)
    sensitivity.add_argument("--dataset", required=True)
    sensitivity.add_argument("--split", choices=["validation", "test"], default="test")
    sensitivity.add_argument("--device", default="cpu")
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
    reach = sub.add_parser(
        "collect-reach",
        help="Generate kinematic reach demonstrations (no Isaac, no ultrasound)",
    )
    reach.add_argument("--landmarks-dir", required=True, help="Directory with landmarks.json + Skin.ply")
    reach.add_argument("--output", required=True)
    reach.add_argument("--config")
    reach.add_argument("--episodes", type=int, default=20)
    reach.add_argument("--no-view", action="store_true")
    playback = sub.add_parser("view-hdf5", help="Synchronized ultrasound and TCP playback in Open3D")
    playback.add_argument("--input", help="Run directory or HDF5 file; default: acq_001")
    playback.add_argument("--skin", help="Skin OBJ; default: data/assets/abdphantom/Skin.obj")
    playback.add_argument("--sample-hz", type=float, default=50)
    playback.add_argument("--mesh-poses", help="JSON with mesh-to-world matrices per demo_N")
    playback.add_argument("--mesh-units", choices=("mm", "m"), default="mm")
    args = parser.parse_args()
    if args.command == "view-hdf5":
        from us_dp.dataset_generation.hdf5_viewer import DEFAULT_RUN, DEFAULT_SKIN, view_hdf5

        view_hdf5(args.input or DEFAULT_RUN, args.skin or DEFAULT_SKIN,
                  sample_hz=args.sample_hz, mesh_poses=args.mesh_poses,
                  mesh_units=args.mesh_units)
        return
    if args.command == "synthetic":
        from us_dp.dataset_generation.synthetic import synthetic_episodes

        result = str(synthetic_episodes(args.output, Config.read(args.config), args.episodes))
    elif args.command == "prepare":
        from us_dp.dataset.processing import prepare

        result = prepare(args.raw, args.output, Config.read(args.config), args.spline_policy_root)
    elif args.command == "train":
        from us_dp.training.train import train

        result = str(
            train(
                args.dataset,
                args.output,
                args.device,
                args.spline_policy_root,
                args.epochs,
                args.batch_size,
                use_image_conditioning=False if args.no_image_conditioning else None,
            )
        )
    elif args.command == "evaluate":
        from us_dp.training.train import evaluate

        result = evaluate(
            args.checkpoint,
            args.dataset,
            args.split,
            args.device,
            args.spline_policy_root,
        )
    elif args.command == "compare-image-conditioning":
        from us_dp.training.compare import compare

        result = compare(
            args.checkpoint_a,
            args.checkpoint_b,
            args.dataset,
            args.split,
            args.device,
            args.spline_policy_root,
            args.output,
            args.sim_episodes,
            args.sim_seed,
            args.skip_offline,
        )
    elif args.command == "image-sensitivity":
        from us_dp.training.image_sensitivity import image_sensitivity

        result = image_sensitivity(
            args.checkpoint,
            args.dataset,
            args.split,
            args.device,
            args.spline_policy_root,
        )
    elif args.command == "smoke":
        from us_dp.dataset_generation.synthetic import smoke

        result = smoke(args.output, args.spline_policy_root)
    elif args.command == "import-hdf5":
        from us_dp.dataset.convert import import_hdf5

        result = str(
            import_hdf5(args.input, args.output, json.loads(Path(args.mapping).read_text()))
        )
    elif args.command == "extract-anatomy":
        from us_dp.anatomy_processing.assets import extract_meshes

        result = str(extract_meshes(args.output, args.image))
    elif args.command == "anatomy":
        from us_dp.anatomy_processing.viewer import prepare_anatomy, view_anatomy

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
        from us_dp.anatomy_processing.viewer import view_anatomy

        view_anatomy(args.input, screenshot=args.screenshot, close_after_capture=args.capture_only)
        result = {"input": args.input}
    elif args.command == "collect-reach":
        from us_dp.config import ReachConfig
        from us_dp.dataset_generation.reach_demo import generate_reach_demonstrations

        saved = generate_reach_demonstrations(
            args.landmarks_dir,
            args.output,
            ReachConfig.read(args.config),
            args.episodes,
            visualize=not args.no_view,
        )
        result = {"episodes": len(saved), "output": str(Path(args.output))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
