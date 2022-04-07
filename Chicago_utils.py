import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
from absl import logging
from tensorflow_transform.tf_metadata import schema_utils
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2
from keras.models import Sequential
from keras.layers import Dense


# Feature to scale from 0 to 1
_RANGE_FEATURE_KEYS = ['arrest_y','case_number_y','PERCENT_OF_HOUSING_CROWDED','PERCENT_OF_HOUSING_CROWDED'
               ,'PERCENT_AGED_16__UNEMPLOYED','PERCENT_AGED_25__WITHOUT_HIGH_SCHOOL_DIPLOMA'
               ,'PERCENT_AGED_UNDER_18_OR_OVER_64','PER_CAPITA_INCOME','HARDSHIP_INDEX','number_of_blocks'
               ,'PERCENT_HOUSEHOLDS_BELOW_POVERTY']



# Features with int data type that will be kept as is
_CATEGORICAL_FEATURE_KEYS = [
    'COMMUNITY_AREA_NAME', 'day','month'
]

# Feature to predict
_LABEL_KEY ='case_number_y'

_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

 
def transformed_name(key):
    return key + '_xf'

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
    inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
    Map from string feature key to transformed feature operations.
    """
    outputs = {}

            

    # Scale these feature/s from 0 to 1
    for key in _RANGE_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_0_1(inputs[key])
            
            
            
            

    # One Hot Encoding.
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

        
    # Casting label to float32
    outputs[transformed_name(_LABEL_KEY)] = tf.cast(inputs[_LABEL_KEY], tf.float32)                               

    return outputs

def reader_fn(filenames):
  '''Load compressed dataset
  
  Args:
    filenames - filenames of TFRecords to load

  Returns:
    TFRecordDataset loaded from the filenames
  '''

  # Load the dataset. Specify the compression type since it is saved as `.gz`
  return tf.data.TFRecordDataset(filenames)
  

def _input_fn(file_pattern,
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output,
              num_epochs=None,
              batch_size=32) -> tf.data.Dataset:
  '''Create batches of features and labels from TF Records

  Args:
    file_pattern - List of files or patterns of file paths containing Example records.
    tf_transform_output - transform output graph
    num_epochs - Integer specifying the number of times to read through the dataset. 
            If None, cycles through the dataset forever.
    batch_size - An int representing the number of records to combine in a single batch.

  Returns:
    A dataset of dict elements, (or a tuple of dict elements and label). 
    Each dict maps feature keys to Tensor or SparseTensor objects.
  '''
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())
  
  datast = data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key="case_number_y_xf"),
      tf_transform_output=transformed_feature_spec).repeat()
    
  
  return dataset


def model_builder() -> tf.keras.Model:
  '''
  Builds the model and sets up the hyperparameters to tune.

  Args:
    hp - Keras tuner object

  Returns:
    model with hyperparameters to tune
  '''
    
  # Initialize the Sequential API and start stacking the layers
  model = Sequential()
  model.add(Dense(500, input_dim=109, activation= "relu"))
  # Get the number of units from the Tuner results
  model.add(Dense(units=100, activation= "relu"))
  model.add(Dense(units=50, activation= "relu"))
  model.add(Dense(1))
  # Get the learning rate from the Tuner results

  # Setup model for training
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001)
                ,loss =keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error") 
                ,metrics=["mean_squared_error"])
  model.summary()
  
  return model


def run_fn(fn_args: FnArgs) -> None:
  """Defines and trains the model.
  Args:
    fn_args: Holds args as name/value pairs. Refer here for the complete attributes: 
    https://www.tensorflow.org/tfx/api_docs/python/tfx/components/trainer/fn_args_utils/FnArgs#attributes
  """

  # Callback for TensorBoard
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')
  
  # Load transform output
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
  
  # Create batches of data good for 10 epochs
  train_set = _input_fn(fn_args.train_files,fn_args.data_accessor, tf_transform_output, num_epochs = 20)
  val_set   = _input_fn(fn_args.eval_files,fn_args.data_accessor,tf_transform_output, num_epochs = 20)


  # Build the model
  model = model_builder()

  # Train the model
  model.fit(
      x=train_set,
      validation_data=val_set,
      callbacks=[tensorboard_callback]
      )
  
  # Save the model
  model.save(fn_args.serving_model_dir, save_format='tf')
