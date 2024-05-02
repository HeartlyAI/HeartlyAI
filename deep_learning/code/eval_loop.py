
def run(task, input_channels, experiment, num_classes):
    
    from utils import utils
    import numpy as np

    sampling_frequency=100
    datafolder='../data/ptbxl/'

    modelname = f'fastai_xresnet1d101{"_"+str(input_channels)+"lead" if input_channels!=12 else ""}'

    outputfolder='../output/'

    # Load PTB-XL data
    data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)
    # Preprocess label data
    scp = raw_labels.scp_codes
    counts = {}

    for threshold in range(5, 105, 5):
        indices = [i for i, d in enumerate(scp) if 'NORM' in d and d['NORM'] == threshold]
        if len(indices) == 0:
            continue
        mean_other_scp_likelihoods = sum([sum([v for k,v in d.items() if k!="NORM"]) for i, d in enumerate(scp.iloc[indices])])/len(indices)

        meta = {"num": len(indices), "mean_other_scp_likelihoods": mean_other_scp_likelihoods}
        counts[threshold] = meta

    #print({k: v for k, v in counts.items() if v["num"] > 0})

    #print(raw_labels["scp_codes"].head(10))
    labels = utils.compute_label_aggregations(raw_labels, datafolder,task)

    # if task == "all":
        #print(labels["all_scp"].head(2))
        
    # elif task=="normabnorm":
        #print(labels["normdiagnostic"].head(10))


    # Select relevant data and convert to one-hot
    data, labels, Y, mlb = utils.select_data(data, labels, task, min_samples=0, outputfolder=outputfolder)
    # for i in range(10):
        #print(Y[i])


    # Max input_channels is 12
    # Extract the first input_channels leads from each sample in data
    if isinstance(input_channels, str):
        i_from, i_to = [int(i) for i in input_channels.split("_")]
        data = np.array([d[:,i_from:i_to] for d in data])
        input_shape = [1000, i_to-i_from]

        # Convert to int for use with model
        input_channels = i_to-i_from
    else:
        data = np.array([d[:,:input_channels] for d in data])
        input_shape = [1000, input_channels] # <=== shape of samples, [None, 12] in case of different lengths

    # 1-9 for training 
    X_train = data[labels.strat_fold < 10]
    y_train = Y[labels.strat_fold < 10]
    # 10 for validation
    X_val = data[labels.strat_fold == 10]
    y_val = Y[labels.strat_fold == 10]



    # X_train.shape, y_train.shape, X_val.shape, y_val.shape

    # # Load pretrained model
    # 
    # For loading a pretrained model:
    #    1. specify `modelname` which can be seen in `code/configs/` (e.g. `modelname='fastai_xresnet1d101'`)
    #    2. provide `experiment` to build the path `pretrainedfolder` (here: `custom_exp` refers to the experiment that only extracts wether the sample is normal or abnormal from the SCP-statements)
    #    
    # This returns the pretrained model where the classification is replaced by a random initialized head with the same number of outputs as the number of classes.

    
    from models.fastai_model import fastai_model

    pretrainedfolder = '../output/'+experiment+'/models/'+modelname+'/'
    mpath = '../output/'+experiment+'/models/'+modelname+'/' # <=== path where the finetuned model will be stored
    n_classes_pretrained = num_classes # <=== because we load the model from exp0, this should be fixed because this depends the experiment

    model = fastai_model(
        modelname, 
        num_classes, 
        sampling_frequency, 
        mpath, 
        input_shape=input_shape, 
        input_channels=input_channels,
        pretrainedfolder=pretrainedfolder,
        n_classes_pretrained=n_classes_pretrained, 
        pretrained=True,
        epochs_finetuning=2,
    )

    # model.input_channels

    
    import pickle

    standard_scaler = pickle.load(open('../output/'+experiment+'/data/standard_scaler.pkl', "rb"))

    X_train = utils.apply_standardizer(X_train, standard_scaler)
    X_val = utils.apply_standardizer(X_val, standard_scaler)

    
    # X_train.shape

    
    X_test = X_val[:]
    y_test = y_val[:]
    # X_test.shape

    
    onnx_file_path = mpath + 'models/model.onnx'
    model.to_onnx(X_test, onnx_file_path)


    
    # this function requires ONLY numpy and onnxruntime
    from utils.onnx import onnx_predict
    y_test_pred_onnx = onnx_predict(X_test, onnx_file_path)

    
    optimal_threshold = 0.75

    for i in range(10):
        idxs = [0]#np.where(y_test[i] > optimal_threshold)[0]
        nan_indices = np.argwhere(np.isnan(y_test_pred_onnx[i]))
        #print(f"i={i}, idxs={idxs}, \tactual={y_test[i][idxs]},  onnx= {y_test_pred_onnx[i][idxs]}")#\ttest_pred = {y_test_pred[i][idxs]}")

    optimal_thresholds = [optimal_threshold] * num_classes
    y_test_pred_onnx_binary = utils.apply_thresholds(y_test_pred_onnx, optimal_thresholds)

    
    actual_classes = mlb.inverse_transform(y_test)

    # y_test_pred_binary = utils.apply_thresholds(y_test_pred, optimal_thresholds)
    # predicted_classes = mlb.inverse_transform(y_test_pred_binary)

    predicted_classes_onnx = mlb.inverse_transform(y_test_pred_onnx_binary)
    # predicted_classes_onnx[:10]

    
    onnx_is_norm = ['NORM' in str(a) for a in predicted_classes_onnx]
    # pred_is_norm = ['NORM' in str(a) for a in predicted_classes]
    actual_is_norm = ['NORM' in str(a) for a in actual_classes]
    #print(f"% of NORM in actual_classes: {sum(actual_is_norm)/len(actual_is_norm)}")
    #print(f"% of NORM in onnx predicted classes: {sum(onnx_is_norm)/len(onnx_is_norm)}")
    # #print(f"% of NORM in predicted classes: {sum(pred_is_norm)/len(pred_is_norm)}")

    
    def calculate_norm_accuracy(predicted_is_norm, actual_is_norm):
        correct_predictions = [1 if a==b else 0 for a, b in zip(predicted_is_norm, actual_is_norm)]
        accuracy = sum(correct_predictions) / len(correct_predictions)
        return accuracy
    onnx_accuracy_norm = calculate_norm_accuracy(onnx_is_norm, actual_is_norm)
    # pred_accuracy_norm = calculate_norm_accuracy(pred_is_norm, actual_is_norm)
    # onnx_accuracy_norm#, pred_accuracy_norm

    
    def calculate_norm_specificity_and_sensitivity(predicted_is_norm, actual_is_norm):
        true_positives = sum([1 if a and b else 0 for a, b in zip(predicted_is_norm, actual_is_norm)])
        true_negatives = sum([1 if not a and not b else 0 for a, b in zip(predicted_is_norm, actual_is_norm)])
        false_positives = sum([1 if a and not b else 0 for a, b in zip(predicted_is_norm, actual_is_norm)])
        false_negatives = sum([1 if not a and b else 0 for a, b in zip(predicted_is_norm, actual_is_norm)])
        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
        return sensitivity, specificity

    def calculate_f1_score(predicted_is_norm, actual_is_norm):
        sensitivity, specificity = calculate_norm_specificity_and_sensitivity(predicted_is_norm, actual_is_norm)
        f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity)
        return f1_score

    
    specificity, sensitivity = calculate_norm_specificity_and_sensitivity(onnx_is_norm, actual_is_norm)
    f1_score = calculate_f1_score(onnx_is_norm, actual_is_norm)

    print(f"Accuracy: {onnx_accuracy_norm}")
    print(f"Specificity: {specificity}, Sensitivity: {sensitivity}, F1-Score: {f1_score}")

    # return results in dict
    results = {
        "accuracy": onnx_accuracy_norm,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "f1_score": f1_score
    }
    
    return results




if __name__ == "__main__":
    results = []

    leads = [1, "1_2", 3, 6, 12]
    experiments = [('custom_exp', 'normabnorm', 1)]#, ('exp0', 'all', 71), ('exp1.1.1', 'superdiagnostic', 5)]

    for lead in leads:
        for experiment in experiments:
            print()
            print()
            print(f"Running experiment {experiment[0]} for task {experiment[1]} with {lead} leads")
            result = run(experiment[1], lead, experiment[0], experiment[2])
            result_dict = {
                "leads": lead,
                "experiment": experiment[0],
                "task": experiment[1],
                **result
            }

            results.append(result_dict)

    import pandas as pd

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Pivot the DataFrame to get leads in rows and experiment in columns
    pivot_df = df.pivot(index='leads', columns='task', values='accuracy')

    # Print the DataFrame in a pretty format
    print(pivot_df.to_string())

    # Save the entire df to a csv file
    df.to_csv("results.csv")
    
    # Save the pivot_df to a csv file
    pivot_df.to_csv("pivot_results.csv")

    # Save only the results for the 'custom_exp' to a csv file, only include Leads	Accuracy	Specificity	Sensitivity	F1_Score
    
    custom_exp_df = df[df['experiment'] == 'custom_exp'][['leads', 'accuracy', 'specificity', 'sensitivity', 'f1_score']]
    custom_exp_df.to_csv("custom_exp_results.csv")
            