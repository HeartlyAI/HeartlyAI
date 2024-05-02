import numpy as np
import onnxruntime as ort

def aggregate_predictions(preds, idmap=None):
    '''
    aggregates potentially multiple predictions per sample (can also pass targs for convenience)
    idmap: idmap as returned by TimeSeriesCropsDataset's get_id_mapping
    preds: ordered predictions as returned by learn.get_preds()
    aggregate_fn: function that is used to aggregate multiple predictions per sample (most commonly np.amax or np.mean)
    '''
    # print(f"Preds recieved: {preds}")
    # print(f"Shape of preds recieved: {preds.shape}")

    aggregate_fn = np.amax

    print("aggregating predictions...")
    preds_aggregated = []
    for i in np.unique(idmap):
        preds_local = preds[np.where(idmap==i)[0]]
        preds_aggregated.append(aggregate_fn(preds_local,axis=0))
    return np.array(preds_aggregated)

    
def split_samples_into_overlapping_chunks(X):
    # X should have shape (batch_size, sequence_length, num_leads)
    chunk_length = 250
    overlap_length = chunk_length // 2
    chunks = []
    idmap = []

    for batch_index in range(X.shape[0]):
        for i in range(0, X.shape[1] - chunk_length + 1, overlap_length):
            chunks.append(X[batch_index, i:i+chunk_length])
            idmap.append(batch_index)

    chunks = np.array(chunks)
    # print(f"Shape of chunks: {chunks.shape}")
    idmap = np.array(idmap)
    return chunks, idmap


def onnx_predict(X, onnx_file_path):
    # Expects X to be a list of 2D numpy arrays
    # Where num rows in list is batch_size and 2D array is (sequence_length, num_leads)
    X = np.array([l.astype(np.float32) for l in X])
    # Load the ONNX model
    session = ort.InferenceSession(onnx_file_path)

    # Get the name of the input node
    input_name = session.get_inputs()[0].name

    # Split the samples into overlapping chunks
    Xt, idmap = split_samples_into_overlapping_chunks(np.array(X))
    
    # transpose X from (batch_size, sequence_length, num_leads) to (batch_size, num_leads, sequence_length)
    # print(f"SHape before transpose: {Xt.shape}")
    Xt = np.transpose(Xt, (0, 2, 1))
    # print(f"SHape after transpose: {Xt.shape}")

    # Prepare your input data in dictionary format
    input_dict = {input_name: Xt}

    # Compute predictions
    outputs = session.run(None, input_dict)

    # 'outputs' is a list of predictions. For many models, it will contain a single array.
    predictions = outputs[0]

    # print(f"Shape of predictions before: {predictions.shape}")
    # if any of predictions is < -22, set it to -22
    predictions = np.clip(predictions, -22, 22) # This should fix the overflow problem
    
    predictions = 1 / (1 + np.exp(-predictions)) # this is the sigmoid function, TODO: when predictions output from model are very large and negative, this causes an overflow. I dont think it is a problem for now, but might want to look into it
    # print(f"Shape of predictions after sigmoid: {predictions.shape}")
    
    predictions_aggregated = aggregate_predictions(predictions, idmap)

    return predictions_aggregated