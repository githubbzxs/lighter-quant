import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, field_validator


class PathConfig(BaseModel):
    data_dir: str = "data"
    models_dir: str = "models"
    log_dir: str = "logs"
    cache_dir: str = "data/cache"


class AppConfig(BaseModel):
    random_seed: int = 42
    log_level: str = "INFO"


class BinanceConfig(BaseModel):
    symbol: str = "BTCUSDT"
    depth_limit: int = 50
    snapshot_limit: int = 200
    rest_base: str = "https://fapi.binance.com"
    ws_base: str = "wss://fstream.binance.com"
    stream_interval: str = "100ms"


class LighterConfig(BaseModel):
    base_url: str = "https://mainnet.zklighter.elliot.ai"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    account_index: int = 0
    fee_rate: float = 0.0
    @field_validator("api_key", "api_secret", mode="before")
    def fill_env(cls, v, info):
        env_key = f"LIGHTER_{info.field_name.upper()}"
        return v or os.getenv(env_key)


class DatasetConfig(BaseModel):
    input_paths: List[str] = Field(default_factory=list)
    output_path: str = "data/processed/dataset.pkl"
    sample_interval_ms: int = 100
    top_levels: int = 10
    agg_depths: List[int] = Field(default_factory=lambda: [5, 10])
    lag_steps: List[int] = Field(default_factory=lambda: [1, 5, 10, 50])
    future_horizon: int = 10
    up_threshold: float = 0.0005
    down_threshold: float = -0.0005
    label_mode: str = "triple"


class TrainConfig(BaseModel):
    model_type: str = "random_forest"
    model_params: dict = Field(default_factory=lambda: {"n_estimators": 200, "max_depth": 8, "n_jobs": -1})
    train_ratio: float = 0.8
    model_output: str = "models/orderbook_model.joblib"

class BacktestConfig(BaseModel):
    dataset_path: str = "data/processed/dataset.pkl"
    model_path: str = "models/orderbook_model.joblib"
    p_buy: float = 0.55
    p_sell: float = 0.55
    hold_ticks: int = 20
    stop_loss: float = -0.003
    take_profit: float = 0.003
    slippage: float = 0.0
    fee_rate: float = 0.0
    grid: Optional[dict] = None
    plot_dir: str = "data/plots"


class LiveConfig(BaseModel):
    model_path: str = "models/orderbook_model.joblib"
    max_position: float = 0.01
    max_single_loss: float = -0.002
    max_daily_loss: float = -0.01
    p_buy: float = 0.55
    p_sell: float = 0.55
    hold_ticks: int = 20
    stop_loss: float = -0.003
    take_profit: float = 0.003
    slippage: float = 0.0
    fee_rate: float = 0.0

class Config(BaseModel):
    app: AppConfig = AppConfig()
    paths: PathConfig = PathConfig()
    binance: BinanceConfig = BinanceConfig()
    lighter: LighterConfig = LighterConfig()
    dataset: DatasetConfig = DatasetConfig()
    train: TrainConfig = TrainConfig()
    backtest: BacktestConfig = BacktestConfig()
    live: LiveConfig = LiveConfig()


def _ensure_dirs(cfg: Config) -> None:
    for path in [cfg.paths.data_dir, cfg.paths.models_dir, cfg.paths.log_dir, cfg.paths.cache_dir]:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = Config(**data)
    _ensure_dirs(cfg)
    return cfg
