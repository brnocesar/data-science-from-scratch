__all__ = ["scale", "rescale", "de_mean_vectors", "direction", "directional_variance", 
           "directional_variance_gradient", "first_principal_component", "project",
           "remove_projection_from_vector", "remove_projection", "pca", "transform_vector", "transform"]

from .algebra_linear import vector_mean, distance, magnitude, dot, scalar_multiply, scalar_multiply
from .estatistica import standard_deviation
from .gradiente import gradient_step
import tqdm

def scale(data: list[list]) -> tuple[list, list]:
    """
    Retorna media e desvio padra de cada dimensao.
    Cada vetor de `data` traz as componentes em cada uma das dimensoes.
    """
    
    # considera que todos os vetores tem a mesma dimensao
    dim = len(data[0]) 
    
    # vetor com as medias de cada dimensao de `data`
    means  = vector_mean(data)
    
    # vetor com os desvios-padrao de cada dimensao de `data`
    # pega i-dimensao de cada vetor e calcula desvio-padrao da dimensao
    # faz isso em todas as dimensoes
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]
    
    return means, stdevs

def rescale(data: list[list]) -> list[list]:
    """
    Redimensiona os dados de entrada para que cada dimensao tenha media 0 e desvio padrao 1.
    Deixa a dimensao como esta se o desvio-padrao for zero.
    """

    dim           = len(data[0])         # considera que todos os vetores tem a mesma dimensao
    means, stdevs = scale(data)          # calcula media e desvio-padra de cada dimensao
    rescaled      = [v[:] for v in data] # faz copia dos vetores

    # percorre cada um dos vetores de `data`
    for v in rescaled:
        for i in range(dim):
            
            # se a i-dimensao tem desvio-padrao diferente de zero
            if stdevs[i] > 0:
                # padroniza essa componente do vetor `v`
                v[i] = (v[i] - means[i]) / stdevs[i]

    return rescaled

def de_mean_vectors(data: list[list]) -> list[list]:
    """
    Centraliza os dados para que todas as dimensoes tenham media 0.
    """
    mean = vector_mean(data)
    return [vector_subtract(v, mean) for v in data]

def direction(w: list) -> list:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]

def directional_variance(data: list[list], w: list) -> float:
    """
    Retorna a variacao de v na direcao de w
    """
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)

def directional_variance_gradient(data: list[list], w: list) -> list:
    """
    O gradiente da variacao direcional em relacao a w
    """
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data) 
            for i in range(len(w))]

def first_principal_component(data: list, n: int = 100, step_size = 0.1) -> list:
    """
    Retorna a primeira componente principal
    """

    # comeca com um valor aleatorio
    guess = [1 for _ in data[0]]
    
    with tqdm.trange(n) as t:
        for _ in t:

            # calcula variacao dos dados na direcao do palpite
            dv = directional_variance(data, guess)

            # calcula o gradiente da variacao direcional
            gradient = directional_variance_gradient(data, guess)

            # atualiza o passo, a direcao de maior variacao
            guess = gradient_step(guess, gradient, step_size)

            # atualiza a barra de progresso
            t.set_description(f"dv: {dv:.3f}")
    
    return direction(guess)

def project(v: list, w: list) -> list:
    """
    Retorna a projecao de v sobre w
    """
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)

def remove_projection_from_vector(v: list, w: list) -> list:
    """
    Projeta v em w e subtrai o resultado de v
    """
    return vector_subtract(v, project(v, w))

def remove_projection(data: list[list], w: list) -> list[list]:
    """
    Projeta cada ponto em w e subtrai o resultado de cada ponto
    """
    return [remove_projection_from_vector(v, w) for v in data]

def pca(data: list[list], num_components: int) -> list[list]:
    components = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)

    return components

def transform_vector(v: list, components: list[list]) -> list:
    return [dot(v, w) for w in components]

dados_transformados = [transform_vector(v, components) for v in dados_centralizados]

def transform(data: list[list], components: list[list]) -> list[list]:
    return [transform_vector(v, components) for v in data]

