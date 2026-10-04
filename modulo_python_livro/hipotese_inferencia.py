__all__ = ["normal_approximation_to_binomial", 
           "normal_probability_below", "normal_probability_above", "normal_probability_between", "normal_probability_outside", 
           "normal_upper_bound", "normal_lower_bound", "normal_two_sided_bounds", 
           "two_sided_p_value", 
           "estimated_parameters", "a_b_test_statistic", 
           "beta_pdf", "beta_two_sided_bound"]

from .probabilidade import normal_cdf
import math
from typing import Tuple
from scipy.stats import beta as beta_dist


# funcao que "define" a aproximacao normal que sera usada
def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """
    Retorna mu e sigma correspondentes a aproximacao de binomial(n,p) como normal
    - como a media de Bernoulli(p) eh p
    - binomial(n,p) eh a soma de n trials de Bernoulli(p)
    - entao:
      - media: mu = np
      - desio padrao: sigma = sqrt(np(1-p))
    """
    mu    = p * n
    sigma = math.sqrt(p * (1 - p) * n)

    return mu, sigma


# a variavel normal (aleatoria) esta abaixo de um limite?
def normal_probability_below(lo: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Probabilidade de N(mu, sigma) ser MENOR ou igual a lo.
    """
    return normal_cdf(lo, mu, sigma)

# esta acima de um limite?
def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Probabilidade de N(mu, sigma) ser MAIOR que lo.
    """
    return 1 - normal_cdf(lo, mu, sigma)

# esta entre dois limites?
def normal_probability_between(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Probabilidade de N(mu, sigma) estar entre lo e hi.
    """
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# esta fora dos limites?
def normal_probability_outside(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Probabilidade de N(mu, sigma) NÃO estar entre lo e hi.
    """
    return 1 - normal_probability_between(lo, hi, mu, sigma)


def normal_upper_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Retorna o z para o qual P(Z <= z) = probability
    """
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Retorna o z para o qual P(Z >= z) = probability
    """
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float, mu: float = 0, sigma: float = 1) -> Tuple[float, float]:
    """
    Retorna os limites simétricos (em torno da média) que contém a probabilidade dada
    """
    tail_probability = (1 - probability) / 2

    # limite superior
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # limite inferior
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound


def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Retorna a probabilidade de ver um resultado tão extremo (ou mais) que x
    """
    if x >= mu:
        # x eh maior do que a media, entao coroa eh qualquer valor maior que x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x eh menor do que a media, entao cara eh qualquer valor menor que x
        return 2 * normal_probability_below(x, mu, sigma)


def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)


def B(alpha, beta):
    """constante normalizadora para que a probabilidade total seja 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:
        return 0
    if x == 0 and alpha < 1:
        return float('inf')
    if x == 1 and beta < 1:
        return float('inf')
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)


def beta_two_sided_bound(probability, alpha, beta):
    tail_probability = (1 - probability) / 2
    lower_bound      = beta_dist.ppf(tail_probability, alpha, beta)
    upper_bound      = beta_dist.ppf(1 - tail_probability, alpha, beta)
    return lower_bound, upper_bound

