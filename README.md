# forensicface

## Instalação

A instalação em ambiente virtual Python é altamente recomendada.
A partir da versão 0.5.1, a versão mínima do Python é 3.13.

``` sh
pip install forensicface
```

## Documentação

- Docs: https://rafribeiro.github.io/forensicface/
- Tutoriais e exemplos: notebooks em `nbs/` (em desenvolvimento)
- Referência de API: https://rafribeiro.github.io/forensicface/api.html

## Como utilizar

Importação da classe ForensicFace:

``` python
from forensicface.app import ForensicFace
```

Instanciamento do ForensicFace:

``` python
ff = ForensicFace(det_size=320, use_gpu=True, extended=True)
```

``` console
[ForensicFace] Initialized with configuration:
                loaded_models=['sepaelv2']
                modules=['detection', 'headpose', 'genderage', 'cr_fiqa']
                det_size=(256, 256)
                session_providers=all models use CUDAExecutionProvider
```


## Processamento básico de imagens

Obter pontos de referência, distância interpupilar, representação
vetorial, a face alinhada com dimensão fixa (112x112), estimativas de
sexo, idade, pose (*pitch*, *yaw*, *roll*) e qualidade. Opcionalmente, é
possível anotar a face alinhada com os pontos de referência utilizados
no alinhamento (parâmetro `draw_kypoints`).

``` python
results = ff.process_image("obama2.png", draw_keypoints=True, single_face=True)
results.keys()
```

    dict_keys(['keypoints', 'ipd', 'embedding', 'norm', 'bbox', 'det_score', 'aligned_face', 'gender', 'age', 'pitch', 'yaw', 'roll', 'fiqa_score'])

``` python
plt.imshow(results["aligned_face"])
```

![](index_files/figure-commonmark/cell-5-output-1.png)

Comparar duas imagens faciais e obter o escore de similaridade.

``` python
ff.compare("obama.png", "obama2.png")
```

    0.8556093

Agregar embeddings de duas imagens faciais em uma única representação,
com ponderação por qualidade

``` python
agg = ff.aggregate_from_images(["obama.png", "obama2.png"], quality_weight=True)
agg.shape
```

    (512,)

## Extração de embeddings em lote

Ao utilizar GPU, é possível acelerar o processamento de múltiplas imagens
passando um lote de imagens simultaneamente para o modelo de extração de
embeddings. Para isso, a partir da versão 0.7.0 há métodos para:  
- apenas detectar e alinhar as faces: `ff.align_only()`  
- extrair embeddings em lote de imagens já alinhadas: `ff._compute_embeddings_batch()`  
- processamento completo de imagens, com extração das embeddings em lote: `ff.process_images_batch()`  

## Estimativa de qualidade CR-FIQA

Estimativa de qualidade pelo método
[CR-FIQA](https://github.com/fdbtrs/CR-FIQA)

Para desabilitar, instancie o forensicface com a opção `extended=False`:

`ff = ForensicFace(extended=False)`

Obs.: a opção `extended=False` também desabilita as estimativas de
sexo, idade e pose.

``` python
good = ff.process_image("001_frontal.jpg")
bad = ff.process_image("001_cam1_1.jpg")
good["fiqa_score"], bad["fiqa_score"]
```

    (2.3786173, 1.4386057)

## Novo layout de pastas a partir da versão 0.7.0
A partir da versão 0.7.0, os arquivos dos modelos pré-treinados são organizados por 
**tipo** em quatro pastas sob `~/.forensicface/models/`:

| Tipo | Caminho |
|---|---|
| Detecção (SCRFD)   | `~/.forensicface/models/detection/det_10g.onnx` |
| Atributos — pose   | `~/.forensicface/models/attributes/1k3d68.onnx` |
| Atributos — sexo/idade | `~/.forensicface/models/attributes/genderage.onnx` |
| Qualidade (CR-FIQA) | `~/.forensicface/models/quality/cr_fiqa_l.onnx` |
| Reconhecimento     | `~/.forensicface/models/recognition/<model_name>/*face*.onnx` |

A estrutura de pastas anterior continua funcionando, mas é recomendado que você
mude para a nova estrutura de pastas. Para auxiliar na migração, foi incluída uma ferramenta para realizar a migração de forma automática:  

`python -m forensicface.tools.migrate_shared` move arquivos para a nova estrutura e
remove as cópias desnecessárias, liberando espaço em disco.  
``` sh
# Dry-run (default): mostra o que seria feito, não toca em nada
python -m forensicface.tools.migrate_shared

# Aplica de fato
python -m forensicface.tools.migrate_shared --apply --yes

# Modelos em diretório customizado
python -m forensicface.tools.migrate_shared --models-root /path/to/models
```

## Crédito dos modelos utilizados

- Detecção, sexo (M/F), idade e pose (pitch, yaw, roll):
  [insightface](https://github.com/deepinsight/insightface)

- Reconhecimento: [adaface](https://github.com/mk-minchul/AdaFace)

- Estimativa de qualidade: [CR-FIQA](https://github.com/fdbtrs/CR-FIQA)

## Notas de versão

v.0.7.0:
- Adicionado suporte a extração de embeddings em lote
- Layout das pastas dos modelos pré-treinados otimizado
- Incluída ferramenta para migração para novo layout de pastas de modelos

v.0.6.0:
- Adicionado suporte ao modelo KPRPE ViT / Adaface / Webface12M, sob o alias `sepaelv6`.
Crédito do modelo: https://github.com/mk-minchul/CVLface

v.0.5.1:
- A dependência do pacote `insightface` foi removida.  
- O diretório padrão dos modelos mudou para `~/.forensicface/models`.  
- É possível especificar outro diretório raiz para os modelos  
    na inicialização do forensicface (parâmetro `models_root`)
- CUDA e CuDNN são instalados automaticamente no ambiente virtual  
    onde o forensicface for instalado.  