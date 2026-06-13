from concise_cot.train import build_sft_config


class DummySFTConfig:
    def __init__(self, output_dir: str, max_length: int):
        self.output_dir = output_dir
        self.max_length = max_length


def test_build_sft_config_maps_max_seq_length_to_max_length() -> None:
    config = build_sft_config(
        DummySFTConfig,
        output_dir="out",
        max_seq_length=128,
        unsupported_field=True,
    )

    assert config.output_dir == "out"
    assert config.max_length == 128
