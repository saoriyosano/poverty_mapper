import tensorflow as tf
from batchers import dataset_constants

IR = ['NIR', 'SWIR1', 'SWIR2']
RGB = ['RED', 'GREEN', 'BLUE']

def augment(ex):
    '''Performs image augmentation (random flips + levels adjustments).
    Does not perform level adjustments on NL band(s).

    Args
    - ex: dict {'images': img, ...}
        - img: tf.Tensor, shape [H, W, C], type float32
            NL band depends on self.ls_bands and self.nl_band

    Returns: ex, with img replaced with an augmented image
    '''
    img = ex['images']

    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    img = self.augment_levels(img)

    ex['images'] = img
    return ex

def process_tfrecords(ex):
    '''
    Args
    - example_proto: a tf.train.Example protobuf

    Returns: dict {'images': img, 'labels': label, 'locs': loc, 'years': year, ...}
    - img: tf.Tensor, shape [224, 224, C], type float32
        - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS]
    - label: tf.Tensor, scalar or shape [2], type float32
        - not returned if both self.label_name and self.nl_label are None
        - [label, nl_label] (shape [2]) if self.label_name and self.nl_label are both not None
        - otherwise, is a scalar tf.Tensor containing the single label
    - loc: tf.Tensor, shape [2], type float32, order is [lat, lon]
    - year: tf.Tensor, scalar, type int32
        - default value of -1 if 'year' is not a key in the protobuf
    - may include other keys if self.scalar_features is not None
    '''
    img_bands = IR  # CHANGE THIS!!!!

    scalar_float_keys = ['lat', 'lon', 'year', 'wealthpooled']
    
    keys_to_features = {}
    for band in img_bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[255**2], dtype=tf.float32)
    for key in scalar_float_keys:
        keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    # if self.scalar_features is not None:
    #     for key, dtype in self.scalar_features.items():
    #         keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=dtype)

    ex = tf.io.parse_single_example(ex, features=keys_to_features)

    
    loc = tf.stack([ex['lat'], ex['lon']])
    year = tf.cast(ex.get('year', -1), tf.int32)

    means = dataset_constants._MEANS_DHS
    std_devs = dataset_constants._STD_DEVS_DHS


    # # for each band, reshape to (255, 255) and crop to (224, 224)
    # # then subtract mean and divide by std dev
    for band in img_bands:
        ex[band].set_shape([255 * 255])
        ex[band] = tf.reshape(ex[band], [255, 255])[15:-16, 15:-16]
    #     ex[band] = (ex[band] - means[band]) / std_devs[band]
    img = tf.stack([ex[band] for band in img_bands], axis=2)

    label = ex.get('wealthpooled', float('nan'))

    # result = {'images': img, 'locs': loc, 'years': year, 'wealthpooled': label}
    result = {'images': img, 'y': label}
    return result
