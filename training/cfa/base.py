"""CFA-specific data boundary. Adapters never register model parameters."""
from abc import ABC, abstractmethod


class CFAAdapter(ABC):
    mode: str
    condition_channels: int
    encoder_channels: int
    tile_size: int
    required_columns: tuple
    batch_columns: tuple
    debug_checks: bool
    samples_per_light: int
    lowlight_apply_constraint: bool

    def validate_images(self, examples, resolution):
        for key in self.required_columns:
            for image in examples[key]:
                if image.size != (resolution, resolution):
                    raise ValueError(
                        f'{key}: expected pre-aligned {resolution}px patches, got {image.size}; '
                        'do not resize CFA mosaics'
                    )

    @abstractmethod
    def pattern_mask(self):
        pass

    @abstractmethod
    def create_reconstructor(self, pattern_mask=None):
        pass

    @abstractmethod
    def preprocess(self, examples, *, resolution, lowlight_simulator, base_transform):
        pass

    @abstractmethod
    def make_conditions(self, batch, dtype=None):
        pass

    @abstractmethod
    def visualization_inputs(self, batch, device):
        """Return conditions, CFA RGB view, and optional W view."""
        pass

    def after_model_init(self, model):
        """Preserve mode-specific startup checks at the original call site."""
