# scikit-learn ライブラリの読み込み
from sklearn import datasets
from sklearn.metrics import classification_report

# 手書き文字セットを読み込む
digits = datasets.load_digits()

# どのようなデータか、確認してみる
import matplotlib.pyplot as plt
plt.matshow(digits.images[5], cmap="Greys")
#plt.show()

# 画像データを配列にしたもの(numpy.ndarray型)
X = digits.data

# 画像データに対する数字(numpy.ndarray型)。ラベルと言う
y = digits.target

# 訓練データとテストデータに分ける
# 訓練データ ：偶数行
X_train, y_train = X[0::2], y[0::2]
# テストデータ：奇数行
X_test, y_test = X[1::2], y[1::2]

# 学習器の作成。SVMというアルゴリズムを選択
from sklearn import svm
clf = svm.SVC(gamma=0.001)

# 訓練データとラベルで学習
clf.fit(X_train, y_train)

# テストデータで試した正解率を返す
accuracy = clf.score(X_test, y_test)
print(f"正解率{accuracy}")

# 学習済モデルを使ってテストデータを分類した結果を返す
predicted = clf.predict(X_test)

# 詳しいレポート
# precision(適合率): 選択した正解/選択した集合
# recall(再現率) : 選択した正解/全体の正解
# F-score(F値) : 適合率と再現率はトレードオフの関係にあるため
print("classification report")
print(classification_report(y_test, predicted))