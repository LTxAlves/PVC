# Princípios de Visão Computacional


## Projeto

Projeto para Princípios de Visão Computacional, visando o uso da biblioteca OpenCV para calibração de câmeras e obtenção de medidas reais a partir de parâmetros de calibração.

## Licensa

Projeto com licensa de código aberto, disponível para livre uso e distribuição.

## Iniciando

Estas instruções vão mostrar os requisitos básicos para utilização do programa além de como utilizá-lo.

### Pré-requisitos

* Python3 (elaborado em versão 3.7.4. Não testado para versões inferiores)
* OpenCV (versão 3.2 ou superior)
* NumPy (elaborado em versão 1.17.0. Não testado para versões inferiores)
* Submódulo xml.etree.ElementTree
* Pandas (elaborado em versão 0.25.1. Não testado para versões inferiores)
* Função `time()` do módulo time
* Funções `remove()` e `exists()` do módulo os e submódulo os.path
* Tipo `datetime` do módulo datetime

## Usando o programa

O programa pode ser utilizado a partir do comando:
```
python3 CameraCalibration.py
```

em seguida, mostrar o padrão de calibração xadrez para realizar o processo de calibração.  
A cada 5 imagens, as janelas "raw" e "undistorted" vão aparecer.  
Pressione a tecla <kbd>Q</kbd> para repetir a calibração, <kbd>R</kbd> para salvar a imagem da janela "raw" e <kbd>U</kbd> para salvar a da janela "undistorted".  
Após 5 repetições, serão gerados os arquivos .xml e será feita uma média e um desvio padrão de cada um, mostrando as janelas "raw" e "undistorted" finais.  
Por fim, clique em qualquer uma das duas janelas para realizar medidas em pixels e em mm - medidas em mm só ocorrem após a calibração com a média dos arquivos .xml.


## Feito com

* [Python3](https://www.python.org/ "Python documentation")
    * [Módulo OS](https://docs.python.org/3/library/os.html "OS module documentation")
    * [Módulo XML](https://docs.python.org/3/library/xml.html "XML module documentation")
    * [Datetime](https://docs.python.org/3/library/datetime.html "Datetime module documentation")
* [OpenCV](https://opencv.org/ "OpenCV documentation")
* [NumPy](https://numpy.org/ "NumPy documentation")
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/ "Pandas documentation")

## Desenvolvedor

* Leonardo Alves - [GitHub](https://github.com/LTxAlves "GitHub de Leonardo Alves")
* Rosana Ribeiro