from gymnasium import Env
from numpy import float32
from numpy.typing import NDArray

type Observation = NDArray[float32]
type DiscreteEnvironment = Env[Observation, int]
