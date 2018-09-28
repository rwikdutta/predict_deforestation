from fastai.conv_learner import *
from fastai.plots import *
from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *
from sklearn.metrics import fbeta_score
import warnings

PATH = "data/"

def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])

def opt_th(preds, targs, start=0.17, end=0.24, step=0.01):
    ths = np.arange(start,end,step)
    idx = np.argmax([fbeta_score(targs, (preds>th), 2, average='samples')
                for th in ths])
    return ths[idx]

metrics=[f2]
f_model = resnet34
label_csv = f'{PATH}train_v2.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)

data = get_data(256)
learn = ConvLearner.pretrained(f_model, data, metrics=metrics)

classes =['agriculture','artisinal_mine','bare_ground','blooming','blow_down','clear','cloudy',
 'conventional_mine','cultivation', 'habitation','haze','partly_cloudy','primary','road',
 'selective_logging','slash_burn','water']

def load_learner(path):
	learn.load(path)

def weighted_predict(tup_pred,ths=0.2):
    preds = []
    danger = ['slash_burn','selective_logging','artisinal_mine','cultivation','agriculture','road']
    #preds = [(2 if label in danger_2 else 1) if pred > th and (label in danger_2 or label in danger_1) else 0\
    #            for label,pred in tup_pred]
    for idx,(label,pred) in enumerate(tup_pred):
        if pred > ths and label in danger:
            preds.append((pred,classes[idx]))
    return preds

def predict(abs_path,size):
    file_1 = abs_path
    trn_tfms,val_tfms=tfms_from_model(f_model,size)
    ds=FilesIndexArrayDataset([file_1],np.array([0]),val_tfms,'')
    dl=DataLoader(ds)
    preds=learn.predict_dl(dl)
    return list(zip(classes,preds[0]))


