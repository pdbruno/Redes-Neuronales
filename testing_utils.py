from typing import Callable, List
import numpy as np
import random

class PosibilidadUmbralAbstract:
  def getN(self, N: int) -> int:
    pass

  def getX(self, x: np.ndarray) -> np.ndarray:
    pass

  def getName(self) -> str:
    pass


class SinUmbral(PosibilidadUmbralAbstract):
  def getN(self, N: int):
    return N

  def getX(self, x: np.ndarray):
    return x

  def getName(self):
    return 'Sin Umbral'

class ConUmbral(PosibilidadUmbralAbstract):
  def getN(self, N: int):
    return N+1

  def getX(self, x: np.ndarray):
    return np.c_[x, np.ones(x.shape[0])]

  def getName(self):
    return 'Con Umbral'

posibilidadesUmbral: List[PosibilidadUmbralAbstract] = [SinUmbral(), ConUmbral()]

class PosibilidadFuncionActivacionAbstract:
  def getG(self, tipoDato: str) -> Callable[[float], float]:
    pass

  def getdGdx(self, tipoDato: str) -> Callable[[np.ndarray], np.ndarray]:
    pass

  def getName(self) -> str:
    pass

class FuncionSigmoidea(PosibilidadFuncionActivacionAbstract):
  def getG(self, tipoDato: str) -> Callable[[float], float]:
    if tipoDato == 'binario':
      return lambda x: 1/(1 + np.exp(-x))
    if tipoDato == 'bipolar':
      return np.tanh
    raise Exception('El tipo de dato ' + tipoDato + ' no existe') 

  def getdGdx(self, tipoDato: str) -> Callable[[np.ndarray], np.ndarray]:
    if tipoDato == 'binario':
      return lambda x: x*(1-x)
    if tipoDato == 'bipolar':
      return lambda x: 1 - np.square(x)
    raise Exception('El tipo de dato ' + tipoDato + ' no existe')

  def getName(self):
    return 'Funcion de Activación Sigmoidea'

class FuncionEscalon(PosibilidadFuncionActivacionAbstract):
  def getG(self, tipoDato: str) -> Callable[[float], float]:
    if tipoDato == 'binario':
      return lambda x: (np.sign(x) + 1)/2
    if tipoDato == 'bipolar':
      return np.sign
    raise Exception('El tipo de dato ' + tipoDato + ' no existe')

  def getdGdx(self, tipoDato: str) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: 1

  def getName(self):
    return 'Funcion de Activación Escalón'

posibilidadesFuncionActivacion: List[PosibilidadFuncionActivacionAbstract] = [FuncionEscalon(), FuncionSigmoidea()]
  
class PosibilidadTipoDatoAbstract:
  def getX(self, x: np.ndarray) -> np.ndarray:
    pass
  
  def getZ(self, z: np.ndarray) -> np.ndarray:
    pass

  def getTipoDato(self) -> str:
    pass

  def getName(self) -> str:
    pass

class TipoDatoBinario(PosibilidadTipoDatoAbstract):
  def getX(self, x: np.ndarray) -> np.ndarray:
    return x
  
  def getZ(self, z: np.ndarray) -> np.ndarray:
    return z

  def getTipoDato(self) -> str:
    return 'binario'    
  
  def getName(self) -> str:
    return 'Tipo de Dato Binario'

class TipoDatoBipolar(PosibilidadTipoDatoAbstract):
  def getX(self, x: np.ndarray) -> np.ndarray:
    return x * 2 - 1
  
  def getZ(self, z: np.ndarray) -> np.ndarray:
    return z * 2 - 1

  def getTipoDato(self) -> str:
    return 'bipolar'
  
  def getName(self) -> str:
    return 'Tipo de Dato Bipolar'

posibilidadesTipoDato: List[PosibilidadTipoDatoAbstract] = [TipoDatoBinario(), TipoDatoBipolar()]

class GeneradorRuidoAbstract:
  def transformX(self, x: np.ndarray, tipoDato: str) -> np.ndarray:
    pass

  def getName(self) -> str:
    pass

class FlipRandomValues(GeneradorRuidoAbstract):

  def transformX(self, x: np.ndarray, tipoDato: str) -> np.ndarray:
    x = x.copy()
    for rowIndex in range(x.shape[0]):
      indexes = list(range(int(x.shape[1] * 0.10))) #flipeo un 10% random de los pixeles de cada imagen
      random.shuffle(indexes)
      for colIndex in indexes:
        x[rowIndex][colIndex] = 1 - x[rowIndex][colIndex] if tipoDato == 'binary' else - x[rowIndex][colIndex]
    return x

  def getName(self) -> str:
    return 'Flip Random Values'

class ScaleValues(GeneradorRuidoAbstract):

  def transformX(self, x: np.ndarray, tipoDato: str) -> np.ndarray:
    if tipoDato == 'binario':
      return x * np.random.uniform( 0.4, 0.6, x.shape)
    if tipoDato == 'bipolar':
      return x * np.random.uniform( -0.1, 0.1, x.shape)
    raise Exception('El tipo de dato ' + tipoDato + ' no existe')

  def getName(self) -> str:
    return 'Scale Values'

generadoresRuido: List[GeneradorRuidoAbstract] = [FlipRandomValues(), ScaleValues()]