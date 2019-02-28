import os
import glob
import tensorflow as tf

GROUND_TRUTH_DIR = os.path.realpath("./ground truth/")
RAINY_IMAGE_DIR = os.path.realpath("./rainy image/")

INDICES_ALL = [
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob(os.path.join(GROUND_TRUTH_DIR, "*.jpg"))
]

IMAGE_SIZE = 384


def _get_input_files(indices=INDICES_ALL):
    return {
        i: glob.glob(os.path.join(RAINY_IMAGE_DIR, "{}_[0-9]*.jpg").format(i))
        for i in indices
    }


def _get_output_files(indices=INDICES_ALL):
    return {
        i: os.path.join(GROUND_TRUTH_DIR, "{}.jpg".format(i)) for i in indices
    }


def dataset(indices=INDICES_ALL):
    """Construct dataset for rainy-image evaluation.

    Args:
    data_dir: Path to the data directory.
    indices: The input-output pairings to return. If None (the default), uses
    indices present in the data directory.

    Returns:
    dataset: Dataset of input-output images.
    """

    fs_in = _get_input_files(indices)
    fs_out = _get_output_files(indices)

    ins = [
        fname for k, v in iter(sorted(fs_in.items()))
        for fname in v if k in indices
    ]

    outs = [v for sublist in [
        [fname] * len(fs_in[k])
        for k, fname in iter(sorted(fs_out.items()))
        if k in indices
    ] for v in sublist]

    def _parse_function(fname_in, fname_out):
        def _decode_resize(fname):
            f = tf.read_file(fname)
            contents = tf.image.decode_jpeg(f)
            resized = tf.image.resize_image_with_crop_or_pad(
                contents, IMAGE_SIZE, IMAGE_SIZE,
            )
            casted = tf.cast(resized, tf.float32)
            return casted

        return _decode_resize(fname_in), _decode_resize(fname_out)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant(ins), tf.constant(outs)),
    ).map(_parse_function)

    return dataset

def write_to_tfrecord(indices):
    fs_in = _get_input_files(indices)
    fs_out = _get_output_files(indices)
    
    writer = tf.python_io.TFRecordWriter('foo.tfrecords')

    def decode_resize(fname):
        f = tf.read_file(fname)
        contents = tf.image.decode_jpeg(f)
        resized = tf.image.resize_image_with_crop_or_pad(
            contents, IMAGE_SIZE, IMAGE_SIZE,
        )
        casted = tf.cast(resized, tf.float32)
        return casted

    for i in indices:
        path_out = fs_out[i]

        paths_in = fs_in[i]

        for path_in in paths_in:
            print('writing file ' + path_in)
            in_raw = decode_resize(path_in).tostring()
            out_raw = decode_resize(path_out).tostring()

            ex = tf.train.Example(features=tf.train.Features(feature={
                'img_in': tf.train.Feature(bytes_list=tf.train.BytesList(value=[in_raw])),
                'img_out': tf.train.Feature(bytes_list=tf.train.BytesList(value=[out_raw])),
            }))

            writer.write(ex.SerializeToString())

    writer.close()
            
if __name__== "__main__":
    write_to_tfrecord(INDICES_ALL)
