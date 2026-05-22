# forensicface


## Install

``` sh
pip install forensicface
```

### Estrutura de pastas dos modelos

A partir desta versão, os modelos são organizados por **tipo** em quatro
pastas sob `~/.forensicface/models/`. Isto evita duplicação dos arquivos
dos modelos de atributos, pose e qualidade.

| Tipo | Caminho |
|---|---|
| Detecção (SCRFD)   | `~/.forensicface/models/detection/det_10g.onnx` |
| Atributos — pose   | `~/.forensicface/models/attributes/1k3d68.onnx` |
| Atributos — sexo/idade | `~/.forensicface/models/attributes/genderage.onnx` |
| Qualidade (CR-FIQA) | `~/.forensicface/models/quality/cr_fiqa_l.onnx` |
| Reconhecimento     | `~/.forensicface/models/recognition/<model_name>/*face*.onnx` |

A estrutura de pastas anterior continua funcionando, mas é recomendado que você
mude para a estrutura nova de pastas. 

## Notas de migração (0.5.1)

- A dependência do pacote `insightface` foi removida.  
- O diretório padrão dos modelos mudou para `~/.forensicface/models`.  
- É possível especificar outro diretório raiz para os modelos  
    na inicialização do forensicface (parâmetro `models_root`)
- CUDA e CuDNN são instalados automaticamente no ambiente virtual  
    onde o forensicface for instalado.  

## Notas de migração (estrutura compartilhada)

- Os modelos de detecção, atributos e qualidade se localizam em
  pastas específicas (`detection/`, `attributes/`, `quality/`) sob
  `models_root`, evitando múltiplas cópias de arquivos onnx.
- Os modelos de reconhecimento ficam em `recognition/<model_name>/`.
- A estrutura antiga (`<model_name>/...`) continua suportada como
  *fallback* — usuários existentes não precisam migrar para a nova
  versão funcionar.

## Documentação

- Tutoriais e exemplos: notebooks em `nbs/`, publicados com Quarto
- Referência de API: gerada via docstrings com `quartodoc`

Build local da documentação:

``` sh
uv sync --extra docs
./scripts/build_docs.sh
```

No `build_docs.sh`, os notebooks são executados localmente antes de gerar o site.
No CI (GitHub Actions), é usado `./scripts/build_docs_ci.sh`, que renderiza sem
executar notebooks novamente.

Saída do site: `_docs/`

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

## Estimativa de qualidade CR-FIQA

Estimativa de qualidade pelo método
[CR-FIQA](https://github.com/fdbtrs/CR-FIQA)

Para desabilitar, instancie o forensicface com a opção extended = False:

`ff = ForensicFace(extended=False)`

Obs.: a opção `extended = False` também desabilita as estimativas de
sexo, idade e pose.

``` python
good = ff.process_image("001_frontal.jpg")
bad = ff.process_image("001_cam1_1.jpg")
good["fiqa_score"], bad["fiqa_score"]
```

    (2.3786173, 1.4386057)

## Crédito dos modelos utilizados

- Detecção, sexo (M/F), idade e pose (pitch, yaw, roll):
  [insightface](https://github.com/deepinsight/insightface)

- Reconhecimento: [adaface](https://github.com/mk-minchul/AdaFace)

- Estimativa de qualidade: [CR-FIQA](https://github.com/fdbtrs/CR-FIQA)
