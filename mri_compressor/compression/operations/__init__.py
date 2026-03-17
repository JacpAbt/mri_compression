"""Individual compression operations."""
from .dead_removal import DeadNeuronRemover
from .neuron_merge import NeuronMerger
from .wanda_pruner import WandaPruner
from .attention_pruner import AttentionHeadPruner
from .depth_pruner import DepthPruner
from .low_rank import LowRankFactorizer
from .reconstructor import LocalReconstructor
from .static_fold import StaticNeuronFolder
from .weight_sharing import WeightSharer
