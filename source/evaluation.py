# source/evaluation.py

from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_paraphrase_detection(y_true, y_pred):
    """Evaluate paraphrase detection using precision, recall, and F1."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 0, 1, 0]
    precision, recall, f1 = evaluate_paraphrase_detection(y_true, y_pred)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
