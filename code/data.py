import os
import numpy as np
import torch
import tensorflow as tf
from pydicom import dcmread
from operators import data_simulation_ray_transform

MAYO_FOLDER = os.path.join( os.getcwd(), 'data', 'mayo_clinic_image', 'full_3mm' )
TEST_RATIO  = 0.2
VAL_RATIO   = 0.1
RANDOM_SEED = 42

def generate_observed_dataset( dataset ):
    assert dataset == 'mayo_clinic_128' or dataset == 'mayo_clinic_512'

    # Numpy random generator
    rng = np.random.default_rng( seed = RANDOM_SEED )

    # Reads filenames with path
    path_to_data = []
    for (dirpath, dirnames, filenames) in os.walk(MAYO_FOLDER):
        path_to_data.extend( [ os.path.join( dirpath, fi ) for fi in filenames ] )
    path_to_data = np.array(path_to_data)

    if path_to_data.size == 0:
        raise ValueError( f'There is no data in {MAYO_FOLDER}. Please see the README.md at {os.path.join( os.getcwd(), 'data', 'mayo_clinic_image' )}.' )

    # Separates dataset into train, test, and validation
    rng.shuffle( path_to_data )
    data_size = path_to_data.shape[0]
    test_size = int( data_size * TEST_RATIO )
    val_size = int( data_size * VAL_RATIO )
    path_to_data_partition = {
        'test' : path_to_data[ :test_size ],
        'val'  : path_to_data[ test_size : test_size + val_size ],
        'train': path_to_data[ test_size + val_size : ]
    }

    # Creates the transform based on the input dataset
    T = data_simulation_ray_transform( dataset )

    for mode in [ 'test', 'val', 'train' ]:
        # Defines the generator for each partition of the dataset
        def gen():
            for fn in path_to_data_partition[mode]:
                # read image from dicom data
                ds = dcmread(fn)
                image = ds.pixel_array
                image = image.astype(np.float32)
                image /= 1000.0
                # generate transformed data with normal 5% noise
                transformed_torch = T( torch.from_numpy(image).float().cuda() )
                transformed = transformed_torch.detach().cpu().numpy()
                noisy = transformed + 0.05 * np.mean( np.abs( transformed ) ) * rng.normal( size = transformed.shape ).astype( np.float32 )
                yield noisy[...,None]
        # Generates dataset
        tf_dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature = tf.TensorSpec(
                shape = ( None, None, 1 ),
                dtype = tf.float32
            )
        )
        # Saves dataset
        tf_dataset.save(
            os.path.join( os.getcwd(), 'data', dataset, 'observed', mode )
        )
        
def generate_F_dataset( dataset, problem, F_ref ):
    for mode in ['train', 'val', 'test']:
        # Loading the observed sinograms dataset
        obs_dataset = get_observed_data( dataset, mode, 32, overall_ratio = 1.0 )
        # Defines the generator
        def gen():
            for b in obs_dataset:
                hist = F_ref( b )
                for h in hist:
                    yield h
        # Generates dataset
        tf_dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature = tf.TensorSpec(
                shape = ( None, ),
                dtype = tf.float32
            )
        )
        # Saves dataset
        tf_dataset.save(
            os.path.join( os.getcwd(), 'data', dataset, 'F_ref', problem, mode )
        )

def get_dataset( dataset, mode, problem, batch_size, F_ref, overall_ratio = 1.0 ):
    assert mode == 'train' or mode == 'val' or mode == 'test'

    # Checks whether the datasets hasn't been generated yet
    dataset_path = os.path.join( os.getcwd(), 'data', dataset )
    if not os.path.exists( os.path.join( dataset_path, 'observed', mode ) ):
        # Generates observed sinograms and saves them
        print('Generating the simulated sinograms...')
        generate_observed_dataset( dataset )
        print('Done!')
    if not os.path.exists( os.path.join( dataset_path, 'F_ref', problem, mode ) ):
        # Generates the functional values for reference
        print('Generating the functional values for reference...')
        generate_F_dataset( dataset, problem, F_ref )
        print('Done!')

    # Loads dataset
    obs_dataset = tf.data.Dataset.load( os.path.join( dataset_path, 'observed', mode ) )
    F_dataset = tf.data.Dataset.load( os.path.join( dataset_path, 'F_ref', problem, mode ) )

    # Zips both datasets together
    Dataset = tf.data.Dataset.zip( obs_dataset, F_dataset )

    # Takes a fraction of the dataset
    examples_to_take = int( Dataset.cardinality().numpy() * overall_ratio )
    Dataset = Dataset.take( examples_to_take )

    if mode == 'train':
        # Shuffles train examples at the beginning of each epoch
        Dataset = Dataset.shuffle( buffer_size = 10 * batch_size, reshuffle_each_iteration = True, seed = RANDOM_SEED )

    # Combines examples on batches
    Dataset = Dataset.batch( batch_size )

    # Prefetch
    Dataset = Dataset.prefetch(tf.data.AUTOTUNE)

    return Dataset

def get_observed_data( dataset, mode, batch_size, overall_ratio = 1.0 ):
    assert mode == 'train' or mode == 'val' or mode == 'test'

    # Checks whether the dataset hasn't been generated yet
    dataset_path = os.path.join( os.getcwd(), 'data', dataset )
    if not os.path.exists( os.path.join( dataset_path, 'observed', mode ) ):
        # Generates observed sinograms and saves them
        print('Generating the simulated sinograms...')
        generate_observed_dataset( dataset )
        print('Done!')

    # Loads dataset
    obs_dataset = tf.data.Dataset.load( os.path.join( dataset_path, 'observed', mode ) )

    # Takes a fraction of the dataset
    examples_to_take = int( obs_dataset.cardinality().numpy() * overall_ratio )
    obs_dataset = obs_dataset.take( examples_to_take )

    if mode == 'train':
        # Shuffles train examples at the beginning of each epoch
        obs_dataset = obs_dataset.shuffle( buffer_size = 10 * batch_size, reshuffle_each_iteration = True, seed = RANDOM_SEED )

    # Combines examples on batches
    obs_dataset = obs_dataset.batch( batch_size )

    # Prefetch
    obs_dataset = obs_dataset.prefetch(tf.data.AUTOTUNE)

    return obs_dataset
