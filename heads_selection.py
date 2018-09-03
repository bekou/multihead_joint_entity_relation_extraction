import utils
import tf_utils
from build_data import build_data
import numpy as np
import eval
import tensorflow as tf
import time

config=build_data()

config.train_id_docs=config.train_id_docs[0:10]

train_data = utils.HeadData(config.train_id_docs, np.arange(len(config.train_id_docs)))
dev_data = utils.HeadData(config.dev_id_docs, np.arange(len(config.dev_id_docs)))
test_data = utils.HeadData(config.test_id_docs, np.arange(len(config.test_id_docs)))


#for doc in train_data.data:
#    print (doc.docId)
#    for tId in range(len(doc.tokens)):
#        print (doc.tokens[tId] + " "+ str(doc.BIOs[tId])+ " "+ str(doc.BIO_ids[tId])+ " "+ str(doc.heads[tId])+ " "+ str(doc.joint_ids[tId])+ " "+ str(doc.ecs[tId])+ " "+ str(doc.ec_ids[tId]))
tf.reset_default_graph()
tf.set_random_seed(1)

utils.printParameters(config)

with tf.Session() as sess:
    embedding_matrix = tf.get_variable('embedding_matrix', shape=config.wordvectors.shape, dtype=tf.float32,
                                       trainable=False).assign(config.wordvectors)
    emb_mtx = sess.run(embedding_matrix)
    
    model = tf_utils.model(config,emb_mtx,sess)

    obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel = model.run()

    train_step = model.get_train_op(obj)

    operations=tf_utils.operations(train_step,obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel)


    sess.run(tf.global_variables_initializer())

    best_score=0
    nepoch_no_imprv = 0  # for early stopping

    for iter in range(config.nepochs):

        model.train(train_data,operations,iter)

        dev_score=model.evaluate(dev_data,operations)

        model.evaluate(test_data, operations)

        if dev_score>=best_score:
            nepoch_no_imprv = 0
            best_score = dev_score

            print ("- Best dev score {} so far in {} epoch".format(dev_score,iter))

        else:
            nepoch_no_imprv += 1
            if nepoch_no_imprv >= config.nepoch_no_imprv:

                print ("- early stopping {} epochs without " \
                                 "improvement".format(nepoch_no_imprv))
                break








