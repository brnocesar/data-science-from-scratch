__all__ = ["difference_quotient", "partial_difference_quotient", "estimate_gradient", "gradient_step",
           "linear_gradient", "partial_difference_quotient", "estimate_gradient", "minibatches"]

from .algebra_linear import distance, vector_add, scalar_multiply, vector_mean
from typing import Callable, TypeVar, List, Iterator
import random

T = TypeVar('T') # permite a insercao de funcoes generica


# estimativa da definicao de derivada para funcao de uma variavel
def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

# derivada parcial
def partial_difference_quotient(f: Callable[[list], float],
                                 v: list,
                                 i: int,
                                 h: float) -> float:
  """Retorna o quociente parcial das diferencas i de f em v"""
  w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)] # adiciona h somente ao elemento i de v

  return (f(w) - f(v)) / h

# estimativa do gradiente
def estimate_gradient(f: Callable[[list], float],
                      v: list,
                      h: float = 0.0001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

# passo na direcao do gradiente
def gradient_step(v: list, 
                  gradient: list,
                  step_size: float) -> list:
    """Move `step_size` na direcao de `gradient` a partir de `v`"""

    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    
    return vector_add(v, step)

# gradiente da funcao de perda para um ponto (x, y)
def linear_gradient(x: float, y: float, theta: list) -> list:
    slope, intercept = theta
    predicted        = slope * x + intercept      # predicao, o que o modelo previu
    error            = (predicted - y)            # erro, diferenca entre o que o modelo previu e o dado real
    squared_error    = error ** 2                 # erro ao quadrado, funcao de perda a ser minimizada
    grad             = [2 * error * x, 2 * error] # gradiente da  funcao de perda
    
    return grad

def partial_difference_quotient(f, v, i, h):
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def minibatches(dataset: list[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[list[T]]:
    """Gera minibatches de tamanho `batch_size` a partir do conjunto de dados"""

    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle:
        # embaralha os dados
        random.shuffle(batch_starts)

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

