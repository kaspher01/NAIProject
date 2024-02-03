from rouge import Rouge


def calculate_metrics(predicted_summary, actual_summary):
    rouge = Rouge()
    scores = rouge.get_scores(predicted_summary, actual_summary)

    return scores[0]['rouge-l']['p'], scores[0]['rouge-l']['r'], scores[0]['rouge-l']['f']
