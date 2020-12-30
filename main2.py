import sklearn.datasets
import sklearn.svm
import numpy
import PIL.Image

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
    print("画像認識AIによる、手書き文字認識の予測結果は",n,"です。")

data = gazouWoSuutini("C:/Users/81908/Documents/Python/number_judge/7.png")

yosoku(data)