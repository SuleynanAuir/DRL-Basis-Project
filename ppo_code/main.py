def main() -> None:
    args = parse_args()
    cfg = default_config(args)
    run_dir, summary = train(cfg)
    print(f"RUN {run_dir}")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
