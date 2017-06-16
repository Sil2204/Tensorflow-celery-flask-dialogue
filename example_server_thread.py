from __future__ import absolute_import, unicode_literals
from __future__ import division
from __future__ import print_function

import math
import os
from os import path, environ
import random
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import time
import json

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
from flask import Flask, Blueprint, abort, jsonify, request, session
import settings
from celery import Celery
from celery.registry import tasks
from celery.signals import before_task_publish,task_success,task_prerun,task_postrun, celeryd_init, worker_process_init
from celery.concurrency import asynpool


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("qt_vocab_size", 50000, "Question vocabulary size.")
tf.app.flags.DEFINE_integer("ans_vocab_size", 50000, "Answer vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/home/sil2204/kbot_dat", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/home/sil2204/kbot_exp", "Training directory.")
tf.app.flags.DEFINE_integer("port", 5000,
                            "server port number.")

FLAGS = tf.app.flags.FLAGS
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
sess = None
model = None
asynpool.PROC_ALIVE_TIMEOUT = 30.0

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
    FLAGS.qt_vocab_size, FLAGS.ans_vocab_size, _buckets,
    FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
    FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
    forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

app = Flask(__name__)
app.config.from_object(settings)

def make_celery(app):
  celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
  celery.conf.update(app.config)
  TaskBase = celery.Task
  class ContextTask(TaskBase):
    abstract = True      
    def __call__(self, *args, **kwargs):
      with app.app_context():  
        return TaskBase.__call__(self, *args, **kwargs) 
  celery.Task = ContextTask
  return celery

celery = make_celery(app)

@worker_process_init.connect
def init_worker(**kwargs):
  global sess
  global model
  sess = tf.Session()
  model = create_model(sess, True)
  return sess, model

@celery.task(name="tasks.dialogue_add", bind=True)
def translate_add(self,sentence):  
  global sess
  global model
  #with tf.Session() as sess:
    # Create model and load parameters.
  #model = create_model(self.sess, True)
  model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
  qt_vocab_path = os.path.join(FLAGS.data_dir,
                               "vocab%d.qt" % FLAGS.qt_vocab_size)
  ans_vocab_path = os.path.join(FLAGS.data_dir,
                               "vocab%d.ans" % FLAGS.ans_vocab_size)
  qt_vocab, _ = data_utils.initialize_vocabulary(qt_vocab_path)
  _, rev_ans_vocab = data_utils.initialize_vocabulary(ans_vocab_path)

    # Get token-ids for the input sentence.
  token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), qt_vocab)
    # Which bucket does it belong to?
  bucket_id = min([b for b in xrange(len(_buckets))
                   if _buckets[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the sentence to the model.
  encoder_inputs, decoder_inputs, target_weights = model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
  _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
  if data_utils.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    # Print out answer sentence corresponding to outputs.
  result = " ".join([tf.compat.as_str(rev_ans_vocab[output]) for output in outputs])
  print("Server sent data:%s" % result)
  return result

@app.route("/test")
def translate(sentence="test"):
  sentence = str(request.args.get("sentence", sentence))
  res = dialogue_add.delay(sentence)
  context = {"id": res.task_id, "sentence": sentence}
  result = "add((sentence):{})".format(context['sentence'])
  goto = "{}".format(context['id'])
  return jsonify(result=result, goto=goto)

@app.route("/test/result/<task_id>")
def show_result(task_id):
  retval = str(dialogue_add.AsyncResult(task_id).get())
  print(retval.decode('utf-8'))
  retval_decode = retval.decode('utf-8')
  context = {"translate_result": retval_decode}
  result = "{}".format(context['translate_result'])
  return jsonify(result=result)


if __name__ == "__main__":
  port = int(environ.get("PORT", FLAGS.port))
  app.run(host='0.0.0.0', port=port, debug=True)
