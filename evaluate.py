from argparse import ArgumentParser, Namespace
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
import torch

from src.model.net import HarmonicCNN
from src.task.pipeline import DataPipeline
from src.task.runner import AutotaggingRunner


def get_config(args: Namespace) -> DictConfig:
    parent_config_dir = Path("conf")
    child_config_dir = parent_config_dir / args.dataset
    model_config_dir = child_config_dir / "model"
    pipeline_config_dir = child_config_dir / "pipeline"
    runner_config_dir = child_config_dir / "runner"
    
    config = OmegaConf.create()
    model_config = OmegaConf.load(model_config_dir / f"{args.model}.yaml")
    pipeline_config = OmegaConf.load(pipeline_config_dir / f"{args.pipeline}.yaml")
    runner_config = OmegaConf.load(runner_config_dir / f"{args.runner}.yaml")
    config.update(model=model_config, pipeline=pipeline_config, runner=runner_config)
    return config

def main(args) -> None:
    seed_everything(42)
    config = get_config(args)

    # prepare dataloader
    pipeline = DataPipeline(pipline_config=config.pipeline)

    dataset = pipeline.get_dataset(
                pipeline.dataset_builder,
                config.pipeline.dataset.path,
                args.type,
                config.pipeline.dataset.input_length
    )
    dataloader = pipeline.get_dataloader(
                dataset,
                shuffle=False,
                drop_last=True,
                **pipeline.pipeline_config.dataloader.params,
    )
    model = HarmonicCNN(**config.model.params)
    runner = AutotaggingRunner(model, config.runner)

    checkpoint_path = (
        f"exp/{args.dataset}/{args.model}/{args.runner}/{args.checkpoint}.ckpt"
    )
    state_dict = torch.load(checkpoint_path)
    runner.load_state_dict(state_dict.get("state_dict"))

    trainer = Trainer(
        **config.runner.trainer.params, logger=False, checkpoint_callback=False
    )
    results_path = Path(f"exp/{args.dataset}/{args.model}/{args.runner}/results.json")

    if results_path.exists():
        with open(results_path, mode="r") as io:
            results = json.load(io)

        result = trainer.test(runner, test_dataloaders=dataloader)
        results.update({"checkpoint": args.checkpoint, f"{args.type}": result})

    else:
        results = {}
        result = trainer.test(runner, test_dataloaders=dataloader)
        results.update({"checkpoint": args.checkpoint, f"{args.type}": result})

    with open(
        f"exp/{args.dataset}/{args.model}/{args.runner}/results.json", mode="w"
    ) as io:
        json.dump(results, io, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", default="TEST", type=str, choices=["TRAIN", "VALID", "TEST"])
    parser.add_argument("--model", default="HarmonicCNN", type=str)
    parser.add_argument("--dataset", default="mtat", type=str, choices=["mtat"])
    parser.add_argument("--pipeline", default="pv00", type=str)
    parser.add_argument("--runner", default="rv00", type=str)
    parser.add_argument("--reproduce", default=False, action="store_true")
    parser.add_argument("--checkpoint", default="epoch=37-roc_auc=0.8806-pr_auc=0.3905", type=str)
    args = parser.parse_args()
    main(args)