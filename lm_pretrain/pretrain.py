"""Training loop."""
# basic example of training a network end-to-end
from time import process_time
from pathlib import Path
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from .model_helper import create_model
from .hparam_helpers import hparams_to_str

def pretrain(hparams):
    """Build and train the model as specified in hparams"""

    ckptsdir = hparams.logdir

    # try to create directory
    if not Path(ckptsdir).exists():
        try:
            Path(ckptsdir).mkdir(parents=True)
        except Exception as e:
            print(e)
            raise

    hparam_str = "Hyperparameters:\n"
    hparam_str += hparams_to_str(hparams)
    # write hparams to directory
    if "lm_hparams" in vars(hparams):
        hparam_str += "\nLanguage Model Hyperparameters:\n"
        hparam_str += hparams_to_str(hparams.lm_hparams)
    try:
        hparam_file = Path(ckptsdir, "hparams.txt")
        hparam_file.write_text(hparam_str)
    except FileNotFoundError as e:
        print(e)
        raise

    # build training and eval graphs
    train_tuple = create_model(hparams, tf.contrib.learn.ModeKeys.TRAIN)
    eval_tuple = create_model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    with train_tuple.graph.as_default():
        initializer = tf.global_variables_initializer()

    with eval_tuple.graph.as_default():
        local_initializer = tf.local_variables_initializer()

    # Summary writers
    if hparams.logging:
        summary_writer = tf.summary.FileWriter(hparams.logdir,
                                               train_tuple.graph,
                                               max_queue=25,
                                               flush_secs=30)

    train_tuple.session.run([initializer])

    if "bdrnn_ckpt" in vars(hparams):
        train_tuple.model.saver.restore(train_tuple.session, hparams.bdrnn_ckpt)
    elif "bdlm_ckpt" in vars(hparams):
        if hparams.model == "bdrnn":
            # the bdlm is a subgraph of the bdrnn
            train_tuple.model.bdlm_saver.restore(train_tuple.session, hparams.bdlm_ckpt)
        else:
            train_tuple.model.saver.restore(train_tuple.session, hparams.bdlm_ckpt)

    start_time = process_time()
    # initialize the training dataset
    train_tuple.session.run([train_tuple.iterator.initializer])
    # finalize the graph
    train_tuple.graph.finalize()

    profile_next_step = False
    eval_step = hparams.eval_step
    patience = 0
    max_patience = hparams.max_patience
    best_eval_loss = np.Inf
    best_step = -1
    # Train until the dataset throws an error (at the end of num_epochs)
    while True:
        step_time = []
        try:
            curr_time = process_time()
            # NOTE: To enable profiling, remove the False at this statement
            if False: # profile_next_step and hparams.logging:
                # run profiling
                _, train_loss, global_step, summary = train_tuple.model.train_with_profile(train_tuple.session, summary_writer)
                profile_next_step = False
            else:
                _, train_loss, global_step, summary = train_tuple.model.train(train_tuple.session)
            step_time.append(process_time() - curr_time)

            # write train summaries
            if global_step == 1 and hparams.logging:
                summary_writer.add_summary(summary, global_step)
            if global_step % 20 == 0:
                if hparams.logging:
                    summary_writer.add_summary(summary, global_step)
                print("Step: %d, Training Loss: %4.4f, Avg Sec/Step: %2.2f" % (global_step, train_loss, np.mean(step_time)))

            if global_step % eval_step == 1:
                step_time = []
                profile_next_step = True
                # Do one evaluation
                checkpoint_path = train_tuple.model.saver.save(train_tuple.session,
                                                               ckptsdir+"/model.ckpt",
                                                               global_step=global_step)
                eval_tuple.model.saver.restore(eval_tuple.session, checkpoint_path)
                eval_tuple.session.run([eval_tuple.iterator.initializer, local_initializer])
                while True:
                    try:
                        inp_tuple, eval_probs, eval_loss, eval_acc, _, eval_summary, _ = eval_tuple.model.eval(eval_tuple.session)
                    except tf.errors.OutOfRangeError:
                        print("Step: %d, Eval Loss: %4.4f, Eval Accuracy: %1.4f" % (global_step,
                                                                              eval_loss,
                                                                              eval_acc))
                        if eval_loss < best_eval_loss:
                            patience = 0
                            best_eval_loss = eval_loss
                            best_step = global_step
                            eval_tuple.model.saver.save(eval_tuple.session,
                                                        ckptsdir+"/best_model.ckpt",
                                                        global_step=best_step)
                        else:
                            patience += 1
                            if patience > max_patience:
                                patience = 0
                                max_patience = int(1.2*max_patience)
                                lr = train_tuple.model.learning_rate / 2.
                                print("Halving learning rate: %g, max patience: %d" % (lr, max_patience))
                                train_tuple.model.update_learning_rate(lr)

                        if hparams.logging:
                            summary_writer.add_summary(eval_summary, global_step)

                        break


        except tf.errors.OutOfRangeError:
            print("- End of Trainig -")
            break

    # End of training
    summary_writer.close()
    print("Best validation loss: %3.3f at step %d" % (best_eval_loss, best_step))
    print("Total Training Time: %4.2f" % (process_time() - start_time))
