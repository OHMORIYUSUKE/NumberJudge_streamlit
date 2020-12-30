import sklearn.datasets
import sklearn.svm
import numpy
import PIL.Image
import streamlit as st
from PIL import Image
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('数字判定アプリ')

st.sidebar.subheader('ダウンロードして使ってください。')
for i in range(10):
    st.sidebar.image(f"{i}.png",use_column_width=True)

digits = datasets.load_digits()

st.sidebar.subheader('教師データ')
for i in range(10):
    plt.matshow(digits.images[i], cmap="Greys")
    st.sidebar.pyplot()


uploaded_file = st.file_uploader("数字の画像をアップロードしてください。")

def gazouWoSuutini(filename):
    gazou = PIL.Image.open(filename).convert("L")
    gazou = gazou.resize((8,8),PIL.Image.ANTIALIAS)
    suuti = numpy.asarray(gazou, dtype = float)
    suuti = numpy.floor(16 - 16 * (suuti / 256))
    suuti = suuti.flatten()
    return suuti


def yosoku(data):

    suuji = sklearn.datasets.load_digits()
    ai = sklearn.svm.SVC(gamma = 0.001)
    ai.fit(suuji.data, suuji.target)
    n = ai.predict([data])

    n = n[0]
    st.subheader("数字識別AIによる、手書き数字識別の予測結果は　"+ str(n) +"　です。")


if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        
        data = gazouWoSuutini(uploaded_file)

        st.image(img , caption='判定画像', use_column_width=True)

        yosoku(data)
    except PIL.UnidentifiedImageError as error:
        st.warning('画像以外がアップロードされました。または、アップロードされた画像は認識できない画像です。')