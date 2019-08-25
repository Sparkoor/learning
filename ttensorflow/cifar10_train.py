"""
单GPU版本的图片分类
"""
import time
import tensorflow as tf
import cifar10

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train', """这是注释""")
tf.app.flags.DEFINE_integer('max_steps', 10000, """number""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """执行的设备的位置""")
tf.app.flags.DEFINE_integer('log_frequency', 10, """how""")


def train():
    # todo:这句话啥意识
    with tf.Graph().as_default():
        # todo:啥意思
        global_step = tf.train.get_or_create_global_step()
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()
        # 推测模型
        logits = cifar10.inference(images)
        loss = cifar10.loss(logits, labels)
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """
            logs loss and runtime
            """

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self,
                          run_context,  # pylint: disable=unused-argument
                          run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    example_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)
                    format_str = ('%s: step %d,loss=%.2f (%.1f example/sec);%.3f sec/batch')
                    print(format_str)

        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                               hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                                                      tf.train.NanTensorHook(loss), _LoggerHook()],
                                               config=tf.ConfigProto(
                                                   log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
