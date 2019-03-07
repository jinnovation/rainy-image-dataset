import glob
import logging
import os

import click
import click_log
import tensorflow as tf

class ExitOnExceptionHandler(logging.StreamHandler):
    def __init__(self, critical_levels, *args, **kwargs):
        self.lvls = critical_levels
        super().__init__(*args, **kwargs)

    def emit(self, record):
        if record.levelno in self.lvls:
            raise SystemExit(-1)

logger = logging.getLogger(__name__)
click_log.basic_config(logger)
logger.handlers.append(ExitOnExceptionHandler([logging.CRITICAL]))

tf.enable_eager_execution()

def indices_all(ground_truth_dir):
    return [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(ground_truth_dir, "*.jpg"))
    ]
    

def serialize_example(f_in, f_out, is_strict=True):
    in_contents = f_in.read()
    out_contents = f_out.read()

    in_shape = tf.io.decode_image(in_contents).shape
    out_shape = tf.io.decode_image(out_contents).shape

    if in_shape != out_shape:
        msg = 'Shape mismatch for file pair ({fname_in}, {fname_out}). In: {dim_in}. Out: {dim_out}'.format(
                fname_in=f_in.name,
                fname_out=f_out.name,
                dim_in=in_shape,
                dim_out=out_shape,
        )
        (logger.critical if is_strict else logger.warning)(msg)

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(in_shape[0])]),
        ),
        'image/width': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(in_shape[1])]),
        ),
        'image/in/filename': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[os.path.basename(f_in.name.encode())]),
        ),
        'image/out/filename': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[os.path.basename(f_out.name.encode())]),
        ),
        'image/in/contents': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[in_contents]),
        ),
        'image/out/contents': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[out_contents]),
        ),
    }))

def _get_input_files(rainy_image_dir, indices):
    return {
        i: glob.glob(os.path.join(rainy_image_dir, "{}_[0-9]*.jpg").format(i))
        for i in indices
    }


def _get_output_files(ground_truth_dir, indices):
    return {
        i: os.path.join(ground_truth_dir, "{}.jpg".format(i)) for i in indices
    }

@click.command()
@click.argument('indices', nargs=-1)
@click.option(
    '--strict',
    default=False,
    show_default=True,
    is_flag=True,
    help='When encountering problematic image pairs, whether or not to terminate.',
)
@click.option(
    '-o', '--out',
    default='rain.tfrecords',
    show_default=True,
    help='File name for the output .tfrecords file.',
)
@click.option(
    '--rainy_image_dir',
    default='./rainy image/',
    type=click.Path(exists=True),
    show_default=True,
)
@click.option(
    '--ground_truth_dir',
    default='./ground truth/',
    type=click.Path(exists=True),
    show_default=True,
)
@click_log.simple_verbosity_option(
    logger,
    default='DEBUG',
    show_default=True,
)
def write_to_tfrecord(
        indices,
        strict,
        out,
        ground_truth_dir,
        rainy_image_dir,
):
    indices = list(indices) or indices_all(ground_truth_dir)
    ground_truth_dir = os.path.realpath(ground_truth_dir)
    rainy_image_dir = os.path.realpath(rainy_image_dir)

    fs_in = _get_input_files(rainy_image_dir, indices)
    fs_out = _get_output_files(ground_truth_dir, indices)
    
    def gen_pairs():
        for i in indices:
            path_out = fs_out[i]
            paths_in = fs_in[i]
            for path_in in paths_in:
                yield (path_in, path_out)

    with tf.python_io.TFRecordWriter(out) as w:
        with click.progressbar(
                list(gen_pairs()),
                label='Writing data pairs',
        ) as pairs:
            for path_in, path_out in pairs:
                with tf.gfile.GFile(path_in, 'rb') as f_in, \
                        tf.gfile.GFile(path_out, 'rb') as f_out:
                    w.write(
                        serialize_example(f_in, f_out, is_strict=strict).SerializeToString(),
                    )

if __name__== "__main__":
    write_to_tfrecord()
