import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    dataset: str
    train_path: str
    dev_path: str
    test_path: str
    input_modality: str = "feature"
    feature_dim: int = 1024
    pose_dim: int = 534
    sbert_sim_path: Optional[str] = None


@dataclass
class EncoderConfig:
    type: str = "vanilla"
    num_keys: int = 5
    query_nb: int = 5


@dataclass
class ModelConfig:
    d_model: int = 256
    num_heads: int = 4
    ffn_dim: int = 512
    num_layers: int = 2
    max_len: int = 512
    dropout: float = 0.1
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class LossConfig:
    label_smoothing: float = 0.1
    auxiliary_type: str = "none"
    auxiliary_weight: float = 1.0


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-5
    clip_grad: float = 1.0
    warmup_steps: int = 200
    patience: int = 10


@dataclass
class EvalConfig:
    metrics: list = field(default_factory=lambda: ["bleu", "rouge"])
    beam_sizes: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    length_penalty: float = 0.6


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    run_name: str = "exp"


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    evaluation: EvalConfig
    output: OutputConfig
    project: str = "SignForge"

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        return Config(
            project=raw.get("project", "SignForge"),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(
                **{k: v for k, v in raw["model"].items() if k != "encoder"},
                encoder=EncoderConfig(**raw["model"].get("encoder_params", {}), type=raw["model"].get("encoder", "vanilla")),
            ),
            loss=LossConfig(
                label_smoothing=raw["loss"]["ce"]["label_smoothing"],
                auxiliary_type=raw["loss"]["auxiliary"].get("type", "none"),
                auxiliary_weight=raw["loss"]["auxiliary"].get("weight", 1.0),
            ),
            training=TrainingConfig(**raw["training"]),
            evaluation=EvalConfig(**raw["evaluation"]),
            output=OutputConfig(**raw["output"]),
        )