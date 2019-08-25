import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('name', 'tom', '''this is name''')


def main(args=None):
    print(FLAGS.name)


if __name__ == '__main__':
    tf.app.run()
