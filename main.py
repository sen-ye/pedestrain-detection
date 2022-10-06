import matplotlib.pyplot as plt
from utils import set_seed, get_data
from sklearn.metrics import f1_score, roc_curve, auc, recall_score, precision_score
from sklearn import svm
import pickle


def train_svm(data, label):
    svc = svm.SVC(kernel='linear', C=0.01)
    svc.fit(data, label)
    s = pickle.dumps(svc)
    f = open('svm.model', "wb+")
    f.write(s)
    f.close()


def test(data, label):
    f = open("svm.model", "rb")
    s = f.read()
    model = pickle.loads(s)
    pred = model.predict(data)
    acc = f1_score(label, pred, average='micro')
    recall = recall_score(label, pred)
    precision = precision_score(label, pred)
    print(f"Accuracy: {acc:.4f}, recall: {recall:.4f}, precision: {precision:.4f}")
    y_score = model.decision_function(data)
    fpr, tpr, th = roc_curve(label, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(16, 16))
    lw = 2
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.title('Receiver Operating Characteristic', fontsize=24)
    plt.legend(loc="lower right", fontsize=18)
    plt.savefig("result_roc")
    plt.show()


def main():
    set_seed(0)
    train_data, train_label = get_data("train")
    test_data, test_label = get_data("test")
    train_svm(train_data, train_label)
    test(test_data, test_label)


if __name__ == "__main__":
    main()