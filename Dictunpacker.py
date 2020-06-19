print(predict_dict)

D_final = pd.DataFrame(predict_dict[0].keys())
for folds in predict_dict.keys():
    D_temp = pd.DataFrame.from_dict(predict_dict[folds])
    D_final = pd.concat(D_final, D_temp, axis=1)