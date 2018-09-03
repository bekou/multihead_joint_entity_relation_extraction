import utils
import time
import eval

class model:
    """Set of classes and methods for training the model and computing the ner and head selection loss"""


    def __init__(self,config,emb_mtx,sess):
        """"Initialize data"""
        self.config=config
        self.emb_mtx=emb_mtx
        self.sess=sess

    def getEvaluator(self):
        if self.config.evaluation_method == "strict" and self.config.ner_classes == "BIO":  # the most common metric
            return eval.chunkEvaluator(self.config, ner_chunk_eval="boundaries_type",
                                                 rel_chunk_eval="boundaries_type")
        elif self.config.evaluation_method == "boundaries" and self.config.ner_classes == "BIO":  # s
            return eval.chunkEvaluator(self.config, ner_chunk_eval="boundaries", rel_chunk_eval="boundaries")
        elif self.config.evaluation_method == "relaxed" and self.config.ner_classes == "EC":  # todo
            return eval.relaxedChunkEvaluator(self.config, rel_chunk_eval="boundaries_type")
        else:
            raise ValueError(
                'Valid evaluation methods : "strict" and "boundaries" in "BIO" mode and "relaxed" in "EC" mode .')


    def train(self,train_data,operations,iter):

            loss = 0

            evaluator = self.getEvaluator()
            start_time = time.time()
            for x_train in utils.generator(train_data, operations.m_op, self.config, train=True):
                _, val, predicted_ner, actual_ner, predicted_rel, actual_rel, _, m_train = self.sess.run(
                    [operations.train_step, operations.obj, operations.predicted_op_ner, operations.actual_op_ner, operations.predicted_op_rel, operations.actual_op_rel, operations.score_op_rel,
                     operations.m_op], feed_dict=x_train)  # sess.run(embedding_init, feed_dict={embedding_placeholder: wordvectors})
                
                if self.config.evaluation_method == "relaxed":
                    evaluator.add(predicted_ner, actual_ner, predicted_rel, actual_rel,m_train['BIO'])
                else:
                    evaluator.add(predicted_ner, actual_ner, predicted_rel, actual_rel)

                loss += val

            print('****iter %d****' % (iter))
            print('-------Train-------')
            print('loss: %f ' % (loss))

            if self.config.evaluation_method == "relaxed":
                evaluator.printInfoMacro()
            else:
                evaluator.printInfo()

            elapsed_time = time.time() - start_time
            print("Elapsed train time in sec:" + str(elapsed_time))
            print()



    def evaluate(self,eval_data,operations,set):

        print('-------Evaluate on '+set+'-------')

        evaluator = self.getEvaluator()
        for x_dev in utils.generator(eval_data, operations.m_op, self.config, train=False):
            predicted_ner, actual_ner, predicted_rel, actual_rel, _, m_eval = self.sess.run(
                [operations.predicted_op_ner, operations.actual_op_ner, operations.predicted_op_rel, operations.actual_op_rel, operations.score_op_rel, operations.m_op], feed_dict=x_dev)

            if self.config.evaluation_method == "relaxed":
                evaluator.add(predicted_ner, actual_ner, predicted_rel, actual_rel, m_eval['BIO'])
            else:
                evaluator.add(predicted_ner, actual_ner, predicted_rel, actual_rel)

        if self.config.evaluation_method == "relaxed":
            evaluator.printInfoMacro()
            return evaluator.getMacroF1scores3Classes()[2] #or evaluator.getMacroF1scores()[2] if we want macro f1 optimization over all classes

        else:
            evaluator.printInfo()
            return  evaluator.getChunkedOverallAvgF1()



    def get_train_op(self,obj):
        import tensorflow as tf

        if self.config.optimizer == 'Adam':

            optim = tf.train.AdamOptimizer(self.config.learning_rate)

        elif self.config.optimizer == 'Adagrad':
            optim = tf.train.AdagradOptimizer(self.config.learning_rate)
        elif self.config.optimizer == 'AdadeltaOptimizer':
            optim = tf.train.AdadeltaOptimizer(self.config.learning_rate)
        elif self.config.optimizer == 'GradientDescentOptimizer':
            optim = tf.train.GradientDescentOptimizer(self.config.learning_rate)

        if self.config.gradientClipping == True:

            gvs = optim.compute_gradients(obj)

            new_gvs = self.correctGradients(gvs)

            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in new_gvs]
            train_step = optim.apply_gradients(capped_gvs)


        else:
            train_step = optim.minimize(obj)

        return train_step

    def correctGradients(self,gvs):
        import tensorflow as tf

        new_gvs = []
        for grad, var in gvs:
            # print (grad)
            if grad == None:

                grad = tf.zeros_like(var)

            new_gvs.append((grad, var))
        if len(gvs) != len(new_gvs):
            print("gradient Error")
        return new_gvs

    def broadcasting(self, left, right):
        import tensorflow as tf



        left = tf.transpose(left, perm=[1, 0, 2])
        left = tf.expand_dims(left, 3)

        right = tf.transpose(right, perm=[0, 2, 1])
        right = tf.expand_dims(right, 0)

        B = left + right
        B = tf.transpose(B, perm=[1, 0, 3, 2])

        return B

    def getNerScores(self, lstm_out, n_types=1, dropout_keep_in_prob=1):
        import tensorflow as tf


        u_a = tf.get_variable("u_typ", [self.config.hidden_size_lstm * 2, self.config.hidden_size_n1])  # [128 32]
        v = tf.get_variable("v_typ", [self.config.hidden_size_n1, n_types])  # [32,1] or [32,10]
        b_s = tf.get_variable("b_typ", [self.config.hidden_size_n1])
        b_c = tf.get_variable("b_ctyp", [n_types])

        mul = tf.einsum('aij,jk->aik', lstm_out, u_a)  # [16 348 64] * #[64 32] = [16 348 32]

        sum = mul + b_s
        if self.config.activation=="tanh":
            output = tf.nn.tanh(sum)
        elif self.config.activation=="relu":
            output = tf.nn.relu(sum)

        if self.config.use_dropout==True:
            output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)

        g = tf.einsum('aik,kp->aip', output, v) + b_c


        return g

    def getHeadSelectionScores(self, lstm_out,dropout_keep_in_prob=1):
        import tensorflow as tf

        u_a = tf.get_variable("u_a", [(self.config.hidden_size_lstm * 2) + self.config.label_embeddings_size, self.config.hidden_size_n1])  # [128 32]
        w_a = tf.get_variable("w_a", [(self.config.hidden_size_lstm * 2) + self.config.label_embeddings_size, self.config.hidden_size_n1])  # [128 32]
        v = tf.get_variable("v", [self.config.hidden_size_n1, len(self.config.dataset_set_relations)])  # [32,1] or [32,4]
        b_s = tf.get_variable("b_s", [self.config.hidden_size_n1])



        left = tf.einsum('aij,jk->aik', lstm_out, u_a)  # [16 348 64] * #[64 32] = [16 348 32]
        right = tf.einsum('aij,jk->aik', lstm_out, w_a)  # [16 348 64] * #[64 32] = [16 348 32]



        outer_sum = self.broadcasting(left, right)  # [16 348 348 32]

        outer_sum_bias = outer_sum + b_s


        if self.config.activation=="tanh":
            output = tf.tanh(outer_sum_bias)
        elif self.config.activation=="relu":
            output = tf.nn.relu(outer_sum_bias)


        if self.config.use_dropout==True:
            output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)


        output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)



        g = tf.einsum('aijk,kp->aijp', output, v)



        g = tf.reshape(g, [tf.shape(g)[0], tf.shape(g)[1], tf.shape(g)[2] * len(self.config.dataset_set_relations)])



        return g



    def computeLoss(self,input_rnn, dropout_embedding_keep,dropout_lstm_keep,dropout_lstm_output_keep,
                    seqlen,dropout_fcl_ner_keep,ners_ids, dropout_fcl_rel_keep,is_train,scoring_matrix_gold, reuse = False):

        import tensorflow as tf

        with tf.variable_scope("loss_computation", reuse=reuse):

            if self.config.use_dropout:
                    input_rnn = tf.nn.dropout(input_rnn, keep_prob=dropout_embedding_keep)
                    #input_rnn = tf.Print(input_rnn, [dropout_embedding_keep], 'embedding:  ', summarize=1000)
            for i in range(self.config.num_lstm_layers):
                if self.config.use_dropout and i>0:
                    input_rnn = tf.nn.dropout(input_rnn, keep_prob=dropout_lstm_keep)
                    #input_rnn = tf.Print(input_rnn, [dropout_lstm_keep], 'lstm:  ', summarize=1000)

                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size_lstm)
                # Backward direction cell
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size_lstm)

                lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw_cell,
                    cell_bw=lstm_bw_cell,
                    inputs=input_rnn,
                    sequence_length=seqlen,
                    dtype=tf.float32, scope='BiLSTM' + str(i))

                input_rnn = tf.concat(lstm_out, 2)

                lstm_output = input_rnn

            if self.config.use_dropout:
                lstm_output = tf.nn.dropout(lstm_output, keep_prob=dropout_lstm_output_keep)


            mask = tf.sequence_mask(seqlen, dtype=tf.float32)

            ner_input = lstm_output
            # loss= tf.Print(loss, [tf.shape(loss)], 'shape of loss is:') # same as scoring matrix ie, [1 59 590]
            if self.config.ner_classes == "EC":

                nerScores = self.getNerScores(ner_input, len(self.config.dataset_set_ec_tags),
                                              dropout_keep_in_prob=dropout_fcl_ner_keep)
                label_matrix = tf.get_variable(name="label_embeddings", dtype=tf.float32,
                                               shape=[len(self.config.dataset_set_ec_tags),
                                                      self.config.label_embeddings_size])
            elif self.config.ner_classes == "BIO":

                nerScores = self.getNerScores(ner_input, len(self.config.dataset_set_bio_tags),
                                              dropout_keep_in_prob=dropout_fcl_ner_keep)
                label_matrix = tf.get_variable(name="label_embeddings", dtype=tf.float32,
                                               shape=[len(self.config.dataset_set_bio_tags),
                                                      self.config.label_embeddings_size])

            # nerScores = tf.Print(nerScores, [tf.shape(ners_ids), ners_ids, tf.shape(nerScores)], 'ners_ids:  ', summarize=1000)

            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                nerScores, ners_ids, seqlen)
            if self.config.ner_loss == "crf":

                lossNER = -log_likelihood
                predNers, viterbi_score = tf.contrib.crf.crf_decode(
                    nerScores, transition_params, seqlen)

            elif self.config.ner_loss == "softmax":
                lossNER = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nerScores, labels=ners_ids)

                predNers = tf.cast(tf.arg_max(nerScores, 2), tf.int32)


            if self.config.label_embeddings_size > 0:

                labels = tf.cond(is_train > 0, lambda: ners_ids, lambda: predNers)


                label_embeddings = tf.nn.embedding_lookup(label_matrix, labels)
                rel_input = tf.concat([lstm_output, label_embeddings], axis=2)

            else:

                rel_input = lstm_output


            rel_scores = self.getHeadSelectionScores(rel_input,
                                                     dropout_keep_in_prob=dropout_fcl_rel_keep)


            lossREL = tf.nn.sigmoid_cross_entropy_with_logits(logits=rel_scores, labels=scoring_matrix_gold)
            probas=tf.nn.sigmoid(rel_scores)
            predictedRel = tf.round(probas)

            return lossNER,lossREL,predNers,predictedRel,rel_scores




    def run(self):

        import tensorflow as tf

        # shape = (batch size, max length of sentence, max length of word)
        char_ids = tf.placeholder(tf.int32, shape=[None, None, None])
        is_train = tf.placeholder(tf.int32)

        # shape = (batch_size, max_length of sentence)
        word_lengths = tf.placeholder(tf.int32, shape=[None, None])

        embedding_ids = tf.placeholder(tf.int32, [None, None])  # [ batch_size  *   max_sequence ]

        token_ids = tf.placeholder(tf.int32, [None, None])  # [ batch_size  *   max_sequence ]

        entity_tags_ids = tf.placeholder(tf.int32, [None, None])

        scoring_matrix_gold = tf.placeholder(tf.float32, [None, None, None])  # [ batch_size  *   max_sequence]


        tokens = tf.placeholder(tf.string, [None, None])  # [ batch_size  *   max_sequence]
        BIO = tf.placeholder(tf.string, [None, None])  # [ batch_size  *   max_sequence]
        entity_tags = tf.placeholder(tf.string, [None, None])  # [ batch_size  *   max_sequence]

        # classes = ...
        seqlen = tf.placeholder(tf.int32, [None])  # [ batch_size ]

        doc_ids = tf.placeholder(tf.string, [None])  # [ batch_size ]


        dropout_embedding_keep = tf.placeholder(tf.float32, name="dropout_embedding_keep")
        dropout_lstm_keep = tf.placeholder(tf.float32, name="dropout_lstm_keep")
        dropout_lstm_output_keep = tf.placeholder(tf.float32, name="dropout_lstm_output_keep")
        dropout_fcl_ner_keep = tf.placeholder(tf.float32, name="dropout_fcl_ner_keep")
        dropout_fcl_rel_keep = tf.placeholder(tf.float32, name="dropout_fcl_rel_keep")

        embedding_matrix = tf.get_variable(name="embeddings", shape=self.emb_mtx.shape,
                                           initializer=tf.constant_initializer(self.emb_mtx), trainable=False)


        #####char embeddings

        # 1. get character embeddings

        K = tf.get_variable(name="char_embeddings", dtype=tf.float32,
                            shape=[len(self.config.dataset_set_characters), self.config.char_embeddings_size])
        # shape = (batch, sentence, word, dim of char embeddings)
        char_embeddings = tf.nn.embedding_lookup(K, char_ids)

        # 2. put the time dimension on axis=1 for dynamic_rnn
        s = tf.shape(char_embeddings)  # store old shape


        char_embeddings_reshaped = tf.reshape(char_embeddings, shape=[-1, s[-2], self.config.char_embeddings_size])
        word_lengths_reshaped = tf.reshape(word_lengths, shape=[-1])



        char_hidden_size = self.config.hidden_size_char

        # 3. bi lstm on chars
        cell_fw = tf.contrib.rnn.BasicLSTMCell(char_hidden_size, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(char_hidden_size, state_is_tuple=True)

        _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                              inputs=char_embeddings_reshaped,
                                                                              sequence_length=word_lengths_reshaped,
                                                                              dtype=tf.float32)
        # shape = (batch x sentence, 2 x char_hidden_size)
        output = tf.concat([output_fw, output_bw], axis=-1)

        # shape = (batch, sentence, 2 x char_hidden_size)
        char_rep = tf.reshape(output, shape=[-1, s[1], 2 * char_hidden_size])

        # concat char embeddings

        word_embeddings = tf.nn.embedding_lookup(embedding_matrix, embedding_ids)

        if self.config.use_chars == True:
            input_rnn = tf.concat([word_embeddings, char_rep], axis=-1)

        else:
            input_rnn = word_embeddings

        embeddings_input=input_rnn


        lossNER, lossREL, predicted_entity_tags_ids, predictedRel, rel_scores = self.computeLoss(input_rnn,
                                                                                dropout_embedding_keep,
                                                                                dropout_lstm_keep,
                                                                                dropout_lstm_output_keep, seqlen,
                                                                                dropout_fcl_ner_keep,
                                                                                entity_tags_ids, dropout_fcl_rel_keep,
                                                                                is_train,
                                                                                scoring_matrix_gold,reuse=False)

        obj = tf.reduce_sum(lossNER) + tf.reduce_sum(lossREL)
        #perturb the inputs
        raw_perturb = tf.gradients(obj, embeddings_input)[0]  # [batch, L, dim]
        normalized_per=tf.nn.l2_normalize(raw_perturb, axis=[1, 2])
        perturb =self.config.alpha*tf.sqrt(tf.cast(tf.shape(input_rnn)[2], tf.float32)) * tf.stop_gradient(normalized_per)
        perturb_inputs = embeddings_input + perturb

        lossNER_per, lossREL_per, _, _, _ = self.computeLoss(perturb_inputs,
                                                             dropout_embedding_keep,
                                                             dropout_lstm_keep,
                                                             dropout_lstm_output_keep, seqlen,
                                                             dropout_fcl_ner_keep,
                                                             entity_tags_ids, dropout_fcl_rel_keep,
                                                             is_train,
                                                             scoring_matrix_gold, reuse=True)

        actualRel = tf.round(scoring_matrix_gold)


        if self.config.use_adversarial==True:

            obj+=tf.reduce_sum(lossNER_per)+tf.reduce_sum(lossREL_per)



        m = {}
        m['isTrain'] = is_train
        m['embeddingIds'] = embedding_ids
        m['charIds'] = char_ids
        m['tokensLens'] = word_lengths
        m['entity_tags_ids'] = entity_tags_ids
        m['scoringMatrixGold'] = scoring_matrix_gold
        m['seqlen'] = seqlen
        m['doc_ids'] = doc_ids
        m['tokenIds'] = token_ids
        m['dropout_embedding']=dropout_embedding_keep
        m['dropout_lstm']=dropout_lstm_keep
        m['dropout_lstm_output']=dropout_lstm_output_keep
        m['dropout_fcl_ner']=dropout_fcl_ner_keep
        m['dropout_fcl_rel'] = dropout_fcl_rel_keep
        m['tokens'] = tokens
        m['BIO'] = BIO
        m['entity_tags'] = entity_tags

        return obj, m, predicted_entity_tags_ids, entity_tags_ids, predictedRel, actualRel, rel_scores


class operations():
    def __init__(self,train_step,obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel):

        self.train_step=train_step
        self.obj=obj
        self.m_op = m_op
        self.predicted_op_ner = predicted_op_ner
        self.actual_op_ner = actual_op_ner
        self.predicted_op_rel = predicted_op_rel
        self.actual_op_rel = actual_op_rel
        self.score_op_rel = score_op_rel