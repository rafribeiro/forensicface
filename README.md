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
- Documentação para contribuidores: [`docs/README.md`](docs/README.md)
- Guia para implementar novos modelos: [`docs/extending-models.md`](docs/extending-models.md)

## Como utilizar

Importação da classe ForensicFace:

``` python
from forensicface import ModelSpec
from forensicface.app import ForensicFace
```

Instanciamento do ForensicFace:

``` python
ff = ForensicFace(
    detection=ModelSpec("scrfd", det_size=320),
    pose="insightface-3d68",
    gender="insightface-genderage",
    age="insightface-genderage",
    quality="cr-fiqa",
    embedding="sepaelv2",
    use_gpu=True,
)
```

Também é possível selecionar cada tarefa diretamente. Seletores omitidos
herdam os defaults definidos por `extended`; `None` desabilita uma tarefa
opcional:

``` python
ff = ForensicFace(
    detection=ModelSpec("centerface", score_threshold=0.35),
    pose="insightface-3d68",
    embedding=["sepaelv2", "sepaelv4"],
)
```

`models=[...]` continua sendo o seletor legado de embeddings e não pode ser
combinado com os novos seletores. Componentes ONNX construídos pelo usuário
também podem ser injetados diretamente. `detection=None` é rejeitado;
`embedding=None` permite detecção, alinhamento e atributos sem reconhecimento.

### Retrocompatibilidade

A API anterior permanece compatível na versão 0.8.0. Chamadas que utilizam
`ForensicFace()` sem seletores, `models`, o parâmetro descontinuado `model`,
`extended`, `det_size`, `det_thresh` ou um `backend` construído continuam
seguindo o fluxo legado e preservam seus defaults e formatos de resultado.

Os novos seletores são uma forma adicional e mais granular de configuração.
Para evitar ambiguidades, uma chamada que utilize qualquer seletor novo não
pode combinar `models`, `model` ou `backend`; utilize `embedding` no lugar de
`models`. Os layouts de modelos das versões anteriores também continuam sendo
consultados como fallback.

``` console
[ForensicFace] Initialized with configuration:
                loaded_models=['sepaelv2']
                modules=['detection', 'headpose', 'genderage', 'cr_fiqa']
                det_size=(320, 320)
                session_providers=all models use CUDAExecutionProvider
```


## Processamento básico de imagens

Obter pontos de referência, distância interpupilar, representação
vetorial, a face alinhada com dimensão fixa (112x112), estimativas de
sexo, idade, pose (*pitch*, *yaw*, *roll*) e qualidade. Opcionalmente, é
possível anotar a face alinhada com os pontos de referência utilizados
no alinhamento (parâmetro `draw_keypoints`).

``` python
results = ff.process_image("obama2.png", draw_keypoints=True, single_face=True)
results.keys()
```

    dict_keys(['ipd', 'fiqa_score', 'gender', 'age', 'yaw', 'pitch', 'roll', 'det_score', 'keypoints', 'bbox', 'embedding', 'aligned_face'])

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

- apenas detectar e alinhar as faces: `ff.detect_and_align()`
- extrair embeddings em lote de faces RGB já alinhadas: `ff.process_aligned_faces_batch()`
- processamento completo de imagens, com extração das embeddings em lote: `ff.process_images_batch()`

## Estimativa de qualidade CR-FIQA

Estimativa de qualidade pelo método
[CR-FIQA](https://github.com/fdbtrs/CR-FIQA)

Para desabilitar a qualidade e os demais estimadores opcionais explicitamente:

``` python
ff = ForensicFace(
    detection="scrfd",
    pose=None,
    gender=None,
    age=None,
    quality=None,
    embedding="sepaelv2",
)
```

O parâmetro legado `extended=False` continua disponível por
retrocompatibilidade e também desabilita qualidade, sexo, idade e pose quando
nenhum seletor novo é utilizado.

``` python
good = ff.process_image("001_frontal.jpg")
bad = ff.process_image("001_cam1_1.jpg")
good["fiqa_score"], bad["fiqa_score"]
```

    (2.3786173, 1.4386057)

## Layout de modelos por tarefa e alias

O layout extensível organiza os modelos por **tarefa** e **alias** sob
`~/.forensicface/models/`:

| Tipo | Caminho |
|---|---|
| Detecção (SCRFD)   | `~/.forensicface/models/detection/scrfd/det_10g.onnx` |
| Detecção (CenterFace) | `~/.forensicface/models/detection/centerface/centerface20260722.onnx` |
| Atributos — pose   | `~/.forensicface/models/pose/insightface-3d68/1k3d68.onnx` |
| Atributos — gênero/idade | `~/.forensicface/models/attributes/insightface-genderage/genderage.onnx` |
| Qualidade (CR-FIQA) | `~/.forensicface/models/quality/cr-fiqa/cr_fiqa_l.onnx` |
| Reconhecimento     | `~/.forensicface/models/recognition/<model_name>/*face*.onnx` |

A estrutura original por modelo e o layout compartilhado plano da versão 0.7
continuam sendo reconhecidos como fallback. A ferramenta de migração aceita
ambos e os converte para os diretórios por tarefa e alias:

`python -m forensicface.tools.migrate_shared` move os arquivos e remove somente
cópias com SHA-256 idêntico, liberando espaço em disco.
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

Ainda não lançado:
- Nenhuma alteração registrada até o momento.

v0.8.0:
- Adicionados os seletores por tarefa `detection`, `pose`, `gender`, `age`,
  `quality` e `embedding`, com precedência da configuração explícita sobre o
  preset legado `extended` e os defaults da biblioteca.
- Adicionado `ModelSpec` genérico para selecionar aliases, caminhos de modelos
  e parâmetros específicos de cada implementação, além do suporte à injeção
  direta de componentes construídos pelo usuário.
- Mantido o suporte a múltiplos modelos de embeddings pelo seletor
  `embedding`; `embedding=None` permite executar detecção, alinhamento e
  atributos sem gerar embeddings.
- Adicionado o detector [CenterFace](https://github.com/Star-Clouds/CenterFace)
- Separados os contratos e adaptadores de detecção, pose, sexo/idade,
  qualidade e embeddings. Cada componente passa a controlar seu próprio
  recorte, tamanho de entrada, normalização, sessão e interpretação da saída.
- Adicionado suporte a estimadores conjuntos por capacidades; o modelo atual
  de sexo e idade é executado apenas uma vez quando atende às duas tarefas.
- A pose passou a utilizar internamente valores nomeados de `pitch`, `yaw` e
  `roll`, preservando os campos e arrays da API pública anterior.
- A inferência de qualidade foi separada da extração de embeddings. O fallback
  por falta de memória da GPU divide lotes de qualidade independentemente, sem
  repetir a inferência de embeddings.
- Adicionado layout de modelos por tarefa e alias, com fallback para os layouts
  das versões anteriores, e adaptada a ferramenta
  `forensicface.tools.migrate_shared` para realizar a migração com validação de
  hashes e conflitos.
- Tarefas opcionais desabilitadas explicitamente deixam de adicionar seus
  campos aos resultados; chamadas pela API legada preservam exatamente seus
  formatos anteriores.
- Mantida a retrocompatibilidade com `models`, `model`, `extended`,
  `det_size`, `det_thresh`, `backend` e os layouts anteriores de modelos.
  `models`, `model` e `backend` não podem ser combinados com os novos seletores
  na mesma chamada. `det_size` e `det_thresh` continuam válidos para SCRFD e
  são rejeitados com detectores aos quais não se aplicam.
- `compute_ss_ds()` é mais eficiente no uso de memória, calculando similaridade em blocos.
- `compute_ss_ds()` não retorna mais os nomes dos arquivos envolvidos em cada score, mas os índices dos arrays de nomes.
- Adicionada a customização de cores para a anotação de pontos-chave por meio
  de `colors` e `keypoint_colors`.
- A anotação agora colore o ponto-chave de índice 1 em vermelho e mantém os
  demais em verde, facilitando a inspeção do alinhamento facial.
- Adicionado um guia para implementação de novos modelos em
  [`docs/extending-models.md`](docs/extending-models.md), incluindo contratos,
  convenções de cor e coordenadas, providers, concorrência e testes.

v0.7.2:
- Adicionado `ff.build_mosaic_from_aligned_faces()`: monta mosaicos a partir de faces RGB já alinhadas (`ff.build_mosaic()` continua detectando e alinhando imagens antes de montar o mosaico de imagens alinhadas).
- Adicionada a função utilitária `forensicface.mosaic.build_mosaic_from_aligned_faces()` para criar mosaicos sem instanciar `ForensicFace`.
- Removida a dependência de `imutils`.

v0.7.1:
- Adicionada documentação para desenvolvedores
- Reorganização interna do código - sem mudanças na API pública.

v0.7.0:
- Adicionado suporte a extração de embeddings em lote
- Layout das pastas dos modelos pré-treinados otimizado
- Incluída ferramenta para migração para novo layout de pastas de modelos

v0.6.0:
- Adicionado suporte ao modelo KPRPE ViT / Adaface / Webface12M, sob o alias `sepaelv6`.
Crédito do modelo: https://github.com/mk-minchul/CVLface

v0.5.1:
- A dependência do pacote `insightface` foi removida.  
- O diretório padrão dos modelos mudou para `~/.forensicface/models`.  
- É possível especificar outro diretório raiz para os modelos  
    na inicialização do forensicface (parâmetro `models_root`)
- CUDA e CuDNN são instalados automaticamente no ambiente virtual  
    onde o forensicface for instalado.  
