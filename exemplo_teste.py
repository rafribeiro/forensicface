from math import e
import lir
import seaborn as sns
import pandas as pd
from itertools import combinations
import numpy as np
from tqdm.notebook import tqdm  # para mostrar barra de progresso
#from numba import jit
import pickle
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from forensicface.app import ForensicFace
ff = ForensicFace(model='sepaelv2', det_size=320, use_gpu=True,
                  gpu=0, extended=True, magface=False)

#@jit(nopython=True)
def cosine(x,y):
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

print('Carregando imagens na memória')

import tkinter as tk
from tkinter import filedialog, messagebox

# root = tk.Tk()
# root.withdraw()

file_pathP = filedialog.askopenfilename(title='Selecione a face padrão...');
file_pathQ = filedialog.askopenfilename(title='Selecione a face questionada...');

str_pathP = file_pathP # "nbs/obama.png"
str_pathQ = file_pathQ # 'nbs/obama2.png'
imageP = cv2.imread(str_pathP)
imageQ = cv2.imread(str_pathQ)

# ajuda a visualizar melhor as faces e as marcações, tomando as dimensões pela menor imagem?
if imageP.shape < imageQ.shape:
    height, width = imageP.shape[:2]
    imageQ = cv2.resize(imageQ, (width, height), interpolation = cv2.INTER_CUBIC)
else:
    height, width = imageQ.shape[:2]
    imageP = cv2.resize(imageP, (width, height), interpolation = cv2.INTER_CUBIC)

    
print('Verificando e extraindo features das faces...')
resultado1 = ff.process_image_multiple_faces(imageP);
resultado2 = ff.process_image_multiple_faces(imageQ);

if len(resultado1) == 1:
    print(resultado1[0].keys())
else:
    print("revise a imagem padrão")
    plt.imshow(imageP)
    
if len(resultado2) == 1:
        print(resultado2[0].keys())
else:
    print("revise a imagem questionada")
    plt.imshow(imageQ)
    
print('Exibindo imagens padrão e questionada...')
fig, axs = plt.subplots(2,2);
fig.suptitle('Imagens padrão e questionada alinhadas');

plt.subplot(2,2,1);
plt.title('Padrao alinhada', fontsize=10)
plt.imshow(resultado1[0]["aligned_face"])
plt.subplot(2,2,2)
plt.title('Questionada alinhada')
plt.imshow(resultado2[0]["aligned_face"])
plt.subplot(2,2,3)
plt.title('Padrao')
plt.imshow(cv2.cvtColor(imageP, cv2.COLOR_BGR2RGB))        
plt.subplot(2,2,4)
plt.title('Questionada')       
plt.imshow(cv2.cvtColor(imageQ, cv2.COLOR_BGR2RGB))

plt.show(block='true');  # display it


print('Marcando pontos chaves...')
draw_imgP = imageP.copy()
draw_imgQ = imageQ.copy()
for kp in resultado1[0]['keypoints']:
    draw_imgP = cv2.circle(draw_imgP, center=kp.astype('int'), radius=5, thickness=2, color=(0,0,255))

for kp in resultado2[0]['keypoints']:
    draw_imgQ = cv2.circle(draw_imgQ, center=kp.astype('int'), radius=5, thickness=2, color=(0,0,255))

#new figure
fig, axs = plt.subplots(1,2);
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(draw_imgP, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(draw_imgQ, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show(block='true');


simil = cosine(resultado1[0]["embedding"], resultado2[0]["embedding"])

str = f'O valor de similaridade entre P e Q é: {simil}'
#simil = ff.compare(str_pathP, str_pathQ)
messagebox.showinfo(title='Resultado da comparação', message=str)
#print(f'\n\nO valor de similaridade entre P e Q é: {simil}.')

''' 
fei_images = glob("../../system_share/bases/fei1000_corrigida/imagens/*.jpg")
fei_images.sort()

fig, axs = plt.subplots(3, 5, figsize=(12, 5))
for imgpath, ax in zip(fei_images[:15], axs.flatten()):
    ax.imshow(Image.open(imgpath))
    ax.axis("off")
    ax.set_title(os.path.basename(imgpath))

    from tqdm.notebook import tqdm  # para mostrar barra de progresso
d = []
for img_path in tqdm(fei_images):
    identity = os.path.basename(img_path)[:3]
    ret = ff.process_image_multiple_faces(img_path)
    if len(ret) == 1:
        embedding = ret[0]["embedding"]
        d.append({"imagem": os.path.basename(img_path),
                 "identidade": identity, "embedding": embedding})

        import pandas as pd
df = pd.DataFrame(d)
df

# with open("fei1000_corrigida_sepaelv2.pkl","wb") as f:
#    pickle.dump(df,f)
with open("recursos/fei1000_corrigida_sepaelv2.pkl", "rb") as f:
    df = pickle.load(f)
df


@jit(nopython=True)
def cosine(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))


comb = combinations(list(df.index), 2)
scores = []
# essa é uma maneira muito ineficiente para cálculo dos escores
for idx1, idx2 in tqdm(list(comb)):
    img1, id1, x1 = df.loc[idx1]
    img2, id2, x2 = df.loc[idx2]

    score = cosine(x1, x2)

    scores.append({'imagem1': img1, 'imagem2': img2,
                  'score': score, 'y': int(id1 == id2)})
scores = pd.DataFrame(scores)
scores


sns.kdeplot(data=scores, x='score', hue='y', common_norm=False, clip=(-1, 1))

questionada = "recursos/001_L1.jpg"
padrao = "recursos/001_frontal.jpg"
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(Image.open(questionada))
ax1.set_title("questionada")
ax2.imshow(Image.open(padrao))
ax2.set_title("padrao")

escore_caso = ff.compare(questionada, padrao)

graph = sns.kdeplot(data=scores, x='score', hue='y',
                    common_norm=False, clip=(-1, 1))
graph.axvline(escore_caso, c='r', linestyle='dashed',
              label=f'Escore do caso - {escore_caso:.3f}')
plt.legend()
plt.show()


calib = lir.LogitCalibrator(C=100.)
calib.fit(scores.score.to_numpy(), scores.y.to_numpy())


x = np.linspace(-0.3, 1., num=100, endpoint=True)

lr = calib.transform(np.array([escore_caso]))
logit = 1/(1+e**-(x*calib._logit.coef_[0][0] + calib._logit.intercept_[0]))

plt.plot(x, logit)
plt.vlines(escore_caso, ymin=-0.01, ymax=1.01, color='r',
           linestyle='dashed', label=f'log10 LR = {np.log10(lr[0]):.2f}')
plt.legend()

calib = lir.LogitCalibrator(C=1.)
calib.fit(scores.score.to_numpy(), scores.y.to_numpy())


x = np.linspace(-0.3, 1., num=100, endpoint=True)

lr = calib.transform(np.array([escore_caso]))
logit = 1/(1+e**-(x*calib._logit.coef_[0][0] + calib._logit.intercept_[0]))

plt.plot(x, logit)
plt.vlines(escore_caso, ymin=-0.01, ymax=1.01, color='r',
           linestyle='dashed', label=f'log10 LR = {np.log10(lr[0]):.2f}')
plt.legend()

 '''